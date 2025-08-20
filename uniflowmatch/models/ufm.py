import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

# Only enable flash attention backend
from uniception.models.encoders import ViTEncoderInput, feature_returner_encoder_factory
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from uniception.models.info_sharing import INFO_SHARING_CLASSES, MultiViewTransformerInput
from uniception.models.prediction_heads.adaptors import (
    ConfidenceAdaptor,
    Covariance2DAdaptor,
    FlowAdaptor,
    FlowWithConfidenceAdaptor,
    MaskAdaptor,
)
from uniception.models.prediction_heads.base import AdaptorMap, PredictionHeadInput, PredictionHeadLayeredInput
from uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from uniception.models.prediction_heads.mlp_feature import MLPFeature
from uniception.models.prediction_heads.moge_conv import MoGeConvFeature

from uniflowmatch.models.rn50_encoder import ResNet50
from uniflowmatch.models.base import (
    UFMClassificationRefinementOutput,
    UFMFlowFieldOutput,
    UFMMaskFieldOutput,
    UFMOutputInterface,
    UniFlowMatchModelsBase,
)
from uniflowmatch.models.unet_encoder import UNet
from uniflowmatch.models.utils import get_meshgrid_torch

from uniflowmatch.models.feature_cache import PreallocatedTensorLRU

CLASSNAME_TO_ADAPTOR_CLASS = {
    "FlowWithConfidenceAdaptor": FlowWithConfidenceAdaptor,
    "FlowAdaptor": FlowAdaptor,
    "MaskAdaptor": MaskAdaptor,
    "Covariance2DAdaptor": Covariance2DAdaptor,
    "ConfidenceAdaptor": ConfidenceAdaptor,
}


# dust3r data structure for reducing passing duplicate images through the encoder
def is_symmetrized(gt1, gt2):
    "Function to check if input pairs are symmetrized, i.e., (a, b) and (b, a) always exist in the input"
    x = gt1["instance"]
    y = gt2["instance"]
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])

    return ok


def interleave(tensor1, tensor2):
    "Interleave two tensors along the first dimension (used to avoid redundant encoding for symmetrized pairs)"
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def modify_state_dict(original_state_dict, mappings):
    """
    Modify state dict keys according to replacement mappings

    Args:
        original_state_dict: Loaded checkpoint state dict
        mappings: Dictionary of {old_key_substr: new_key_substr_or_None}

    Returns:
        Modified state dictionary with updated keys
    """
    new_state_dict = {}

    for k, v in original_state_dict.items():
        new_key = None
        skip = False

        # Check for all possible replacements
        for replace_key, replace_value in mappings.items():
            if replace_key in k:
                if replace_value is None:
                    skip = True
                    break  # Skip this key entirely
                else:
                    new_key = k.replace(replace_key, replace_value)
                    break  # Only apply first matching replacement

        if skip:
            continue

        new_state_dict[new_key if new_key is not None else k] = v

    return new_state_dict


class UniFlowMatch(UniFlowMatchModelsBase, PyTorchModelHubMixin):
    """
    UniFlowMatch model.
    """

    def __init__(
        self,
        # Encoder configurations
        encoder_str: str,
        encoder_kwargs: Dict[str, Any],
        cnn_augment: bool = False,
        # Info sharing & output head structure configurations
        info_sharing_and_head_structure: str = "dual+single",  # only dual+single is supported
        # Information sharing configurations
        info_sharing_str: str = "global_attention",
        info_sharing_kwargs: Dict[str, Any] = {},
        # skip-connections between encoder and info-sharing
        encoder_skip_connection: Optional[List[int]] = None,
        info_sharing_skip_connection: Optional[List[int]] = None,
        # Prediction Heads & Adaptors
        head_type: str = "dpt",
        feature_head_kwargs: Dict[str, Any] = {},
        adaptors_kwargs: Dict[str, Any] = {},
        # Load Pretrained Weights
        pretrained_checkpoint_path: Optional[str] = None,
        # Inference Settings
        inference_resolution: Optional[Tuple[int, int]] = (560, 420),  # WH
        *args,
        **kwargs,
    ):
        """
        Initialize the UniFlowMarch Model

        - encoder_str (str): Encoder string
        - encoder_kwargs (Dict[str, Any]): Encoder configurations

        - info_sharing_and_head_structure (str): Info sharing and head structure configurations
            - "dual+single": Dual view info sharing and single view prediction head

        - info_sharing_str (str): Info sharing method
            - "global_attention_transformer": Global attention transformer
        - info_sharing_kwargs (Dict[str, Any]): Info sharing configurations

        """
        UniFlowMatchModelsBase.__init__(self, inference_resolution=inference_resolution, *args, **kwargs)

        PyTorchModelHubMixin.__init__(self)

        # assertion on architectures
        assert info_sharing_and_head_structure == "dual+single", "Only dual+single is supported now"

        # initialize the skip-connections
        self.encoder_skip_connection = encoder_skip_connection
        self.info_sharing_skip_connection = info_sharing_skip_connection

        # initialize encoder
        self.encoder: nn.Module = feature_returner_encoder_factory(encoder_str, **encoder_kwargs)

        self.cnn_augment = cnn_augment
        if self.cnn_augment:
            self.cnn_encoder: nn.Module = ResNet50(pretrained=True, freeze_bn=False)
            self.cnn_feat_proj: nn.Module = nn.Conv2d(1024, self.encoder.enc_embed_dim, kernel_size=1)

        self.cached_features = PreallocatedTensorLRU(max_items=6, feature_shape=(self.encoder.enc_embed_dim, 320//16, 480//16), device="cpu", dtype=torch.float16)

        # initialize info-sharing module
        assert head_type != "linear", "Linear head is not supported, because it have major disadvantage to DPTs"
        self.head_type = head_type

        self.info_sharing: nn.Module = INFO_SHARING_CLASSES[info_sharing_str][1](**info_sharing_kwargs)

        self.head1: nn.Module = self._initialize_prediction_heads(head_type, feature_head_kwargs, adaptors_kwargs)

        # load pretrained weights
        if pretrained_checkpoint_path is not None:
            ckpt = torch.load(pretrained_checkpoint_path, map_location="cpu")

            if "state_dict" in ckpt:
                # we are loading from training checkpoint directly.
                model_state_dict = ckpt["state_dict"]
                model_state_dict = {
                    k[6:]: v for k, v in model_state_dict.items() if k.startswith("model.")
                }  # remove "model." prefix

                model_state_dict = modify_state_dict(
                    model_state_dict, {"feature_matching_proj": None, "encoder.model.mask_token": None}
                )

                self.load_state_dict(model_state_dict, strict=True)
            else:
                model_state_dict = ckpt["model"]

                load_result = self.load_state_dict(model_state_dict, strict=False)
                assert len(load_result.missing_keys) == 0, f"Missing keys: {load_result.missing_keys}"

    def to(self, *args, **kwargs):
        self.cached_features.to(*args, **kwargs)
        super().to(*args, **kwargs)

    @classmethod
    def from_pretrained_ckpt(cls, pretrained_model_name_or_path, strict=True, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu")

            # remove base_pretrained_checkpoint_path from the model args
            if "base_pretrained_checkpoint_path" in ckpt["model_args"]:
                ckpt["model_args"].pop("base_pretrained_checkpoint_path")

            # convert old model args into new definition
            if "img_size" in ckpt["model_args"]:
                # we are loading from a old benchmark checkpoint
                print("Converting from a old benchmark checkpoint")
                model_args = {
                    # Encoder args
                    "encoder_str": ckpt["model_args"]["encoder_str"],
                    "encoder_kwargs": ckpt["model_args"]["encoder_kwargs"],
                    # Info-sharing args
                    "info_sharing_and_head_structure": "dual+single",
                    "info_sharing_str": ckpt["model_args"]["info_sharing_type"],
                    "info_sharing_kwargs": {
                        "name": "info_sharing",
                        "input_embed_dim": ckpt["model_args"]["input_embed_dim"],
                        "num_views": 2,
                        "use_rand_idx_pe_for_non_reference_views": False,
                        "depth": ckpt["model_args"]["num_layers"],
                        "dim": ckpt["model_args"]["transformer_dim"],
                        "num_heads": ckpt["model_args"]["num_heads"],
                        "mlp_ratio": ckpt["model_args"]["mlp_ratio"],
                        "qkv_bias": ckpt["model_args"]["qkv_bias"],
                        "qk_norm": ckpt["model_args"]["qk_norm"],
                        "custom_positional_encoding": ckpt["model_args"]["position_encoding"],
                        "norm_intermediate": ckpt["model_args"]["normalize_intermediate"],
                        "indices": ckpt["model_args"]["returned_intermediate_layers"],
                    },
                    # flow head args
                    "head_type": "dpt",
                    "feature_head_kwargs": ckpt["model_args"]["feature_head_kwargs"],
                    "adaptors_kwargs": ckpt["model_args"]["adaptors_kwargs"],
                }

                if "covocc_feature_head_kwargs" in ckpt["model_args"]:
                    # if the model has a covocc head, we need to convert it to the new format
                    model_args["uncertainty_head_type"] = "dpt"
                    model_args["uncertainty_head_kwargs"] = {
                        "dpt_feature": ckpt["model_args"]["covocc_feature_head_kwargs"]["dpt_feature"],
                        "dpt_processor": ckpt["model_args"]["covocc_feature_head_kwargs"]["dpt_regr_processor"],
                    }
                    model_args["uncertainty_adaptors_kwargs"] = {
                        "flow_cov": ckpt["model_args"]["covocc_adaptors_kwargs"]["flow_cov"]
                    }

                ckpt["model_args"] = model_args

                # Update the old weights into the current format
                ckpt["model"] = modify_state_dict(
                    ckpt["model"],
                    {
                        "covocc_head.dpt_feature": "uncertainty_head.0.0",
                        "covocc_head.dpt_regr_processor": "uncertainty_head.0.1",
                        "covocc_head.dpt_segm_processor": None,
                        "feature_matching_proj": None,
                        "encoder.model.mask_token": None,
                    },
                )

            # remove the ket "pretrained_backbone_checkpoint_path" from the model args
            if "pretrained_backbone_checkpoint_path" in ckpt["model_args"]:
                ckpt["model_args"].pop("pretrained_backbone_checkpoint_path")

            model = cls(**ckpt["model_args"], **kw)
            model.load_state_dict(ckpt["model"], strict=strict)
            return model
        else:
            raise ValueError(f"Pretrained model {pretrained_model_name_or_path} not found.")

    def _initialize_prediction_heads(
        self, head_type: str, feature_head_kwargs: Dict[str, Any], adaptors_kwargs: Dict[str, Any]
    ):
        """
        Initialize prediction heads and adaptors

        Args:
        - head_type (str): Head type, either "dpt" or "linear"
        - feature_head_kwargs (Dict[str, Any]): Feature head configurations
        - adaptors_kwargs (Dict[str, Any]): Adaptors configurations

        Returns:
        - nn.Module: output head + adaptors
        """
        feature_processor: nn.Module
        if head_type == "dpt":
            feature_processor = nn.Sequential(
                DPTFeature(**feature_head_kwargs["dpt_feature"]),
                DPTRegressionProcessor(**feature_head_kwargs["dpt_processor"]),
            )
        elif head_type == "moge_conv":
            feature_processor = MoGeConvFeature(**feature_head_kwargs)
        else:
            raise ValueError(f"Head type {head_type} not supported.")

        adaptors = self._initialize_adaptors(adaptors_kwargs)

        return nn.Sequential(feature_processor, AdaptorMap(*adaptors.values()))

    def _initialize_adaptors(self, adaptors_kwargs: Dict[str, Any]):
        """
        Initialize a dict of adaptors

        Args:
        - adaptors_kwargs (Dict[str, Any]): Adaptors configurations

        Returns:
        - Dict[str, nn.Module]: dict of adaptors, from adaptor's name to the adaptor
        """
        return {
            name: CLASSNAME_TO_ADAPTOR_CLASS[configs["class"]](**configs["kwargs"])
            for name, configs in adaptors_kwargs.items()
        }

    def _encode_and_cache_image_features(self, img: torch.Tensor, labels: List[str], data_norm_type: str):
        """
        A read-through cache for image features. This speeds up inference by reusing previous image-encoder computations. 
        """

        B, C, H, W = img.shape
        patch_size = self.encoder.patch_size
        Hp, Wp = H // patch_size, W // patch_size

        outputs = torch.zeros((B, self.encoder.enc_embed_dim, Hp, Wp), dtype=torch.float16, device=img.device)

        # copy the available features
        remaining_indexes = []

        for index, label in enumerate(labels):
            if not (label in self.cached_features):
                remaining_indexes.append(index)

        # compute unique index of the unique label
        unique_labels = list(set(labels[index] for index in remaining_indexes))

        unique_indices = [
            remaining_indexes[
                [labels[i] for i in remaining_indexes].index(label)
            ]
            for label in unique_labels
        ]

        # compute the features for the remaining features
        if unique_indices:
            remaining_imgs = img[unique_indices]
            encoder_input = ViTEncoderInput(image=remaining_imgs, data_norm_type=data_norm_type)
            output = self.encoder(encoder_input)

            if self.cnn_augment:
                # Re-normalize input image to ImageNet
                prior_mean, prior_std = (
                    IMAGE_NORMALIZATION_DICT[data_norm_type].mean,
                    IMAGE_NORMALIZATION_DICT[data_norm_type].std,
                )
                imgnet_mean, imgnet_std = (
                    IMAGE_NORMALIZATION_DICT["dinov2"].mean,
                    IMAGE_NORMALIZATION_DICT["dinov2"].std,
                )

                scale = (prior_std / imgnet_std).to(remaining_imgs.device)
                offset = (imgnet_mean - prior_mean).to(remaining_imgs.device)

                imgnet_normalized_cat_imgs = remaining_imgs * scale.view(1, 3, 1, 1) + offset.view(1, 3, 1, 1)

                if self.encoder.patch_size == 14:
                    # Interpolate images to be 16/14 larger so that the number of patches stays the same
                    imgnet_normalized_cat_imgs = torch.nn.functional.interpolate(
                        imgnet_normalized_cat_imgs,
                        scale_factor=16 / 14,
                        align_corners=False,
                        mode="bicubic",
                        antialias=True,
                    )

                cnn_feat = self.cnn_encoder(imgnet_normalized_cat_imgs)[16]
                cnn_feat = self.cnn_feat_proj(cnn_feat)

                output[-1].features += cnn_feat

            # store the newly computed features
            for i, index in enumerate(unique_indices):
                self.cached_features.put(labels[index], output[-1].features[i])

            # read-out all features from the cache
            for index, label in enumerate(labels):
                outputs[index] = self.cached_features.get(label)

        return outputs
    

    def _encode_image_pairs(self, img1, img2, data_norm_type):
        "Encode two different batches of images (each batch can have different image shape)"
        if img1.shape[-2:] == img2.shape[-2:]:
            cat_images = torch.cat((img1, img2), dim=0)
            encoder_input = ViTEncoderInput(image=cat_images, data_norm_type=data_norm_type)
            output = self.encoder(encoder_input)

            if self.cnn_augment:
                # Re-normalize input image to ImageNet
                prior_mean, prior_std = (
                    IMAGE_NORMALIZATION_DICT[data_norm_type].mean,
                    IMAGE_NORMALIZATION_DICT[data_norm_type].std,
                )
                imgnet_mean, imgnet_std = (
                    IMAGE_NORMALIZATION_DICT["dinov2"].mean,
                    IMAGE_NORMALIZATION_DICT["dinov2"].std,
                )

                scale = (prior_std / imgnet_std).to(cat_images.device)
                offset = (imgnet_mean - prior_mean).to(cat_images.device)

                imgnet_normalized_cat_imgs = cat_images * scale.view(1, 3, 1, 1) + offset.view(1, 3, 1, 1)

                if self.encoder.patch_size == 14:
                    # Interpolate images to be 16/14 larger so that the number of patches stays the same
                    imgnet_normalized_cat_imgs = torch.nn.functional.interpolate(
                        imgnet_normalized_cat_imgs,
                        scale_factor=16 / 14,
                        align_corners=False,
                        mode="bicubic",
                        antialias=True,
                    )

                cnn_feat = self.cnn_encoder(imgnet_normalized_cat_imgs)[16]
                cnn_feat = self.cnn_feat_proj(cnn_feat)

            out_feats, out2_feats = [], []
            for idx, feat in enumerate(output):
                feat1, feat2 = feat.features.chunk(2, dim=0)

                if (idx == len(output) - 1) and self.cnn_augment:
                    cnn_feat1, cnn_feat2 = cnn_feat.chunk(2, dim=0)
                    feat1 = feat1 + cnn_feat1
                    feat2 = feat2 + cnn_feat2

                out_feats.append(feat1)
                out2_feats.append(feat2)

            return out_feats, out2_feats
        else:
            raise NotImplementedError("Different image shapes are not supported yet")

    def _encode_symmetrized(self, view1, view2, symmetrized=False):
        "Encode image pairs accounting for symmetrization, i.e., (a, b) and (b, a) always exist in the input"
        img1 = view1["img"]
        img2 = view2["img"]

        feat1_list, feat2_list = [], []

        if symmetrized:
            # Computing half of forward pass'
            # modified in conjunction with UFM for not copying the images again.
            # used to be: feat1, feat2 = self._encode_image_pairs(img1[::2], img2[::2], data_norm_type=view1["data_norm_type"])
            # be very carefult with this!!!
            feat1_list_, feat2_list_ = self._encode_image_pairs(
                img1[::2], img2[::2], data_norm_type=view1["data_norm_type"]
            )

            for feat1, feat2 in zip(feat1_list_, feat2_list_):
                feat1, feat2 = interleave(feat1, feat2)
                feat1_list.append(feat1)
                feat2_list.append(feat2)
        else:
            feat1_list, feat2_list = self._encode_image_pairs(img1, img2, data_norm_type=view1["data_norm_type"])

        return feat1_list, feat2_list

    def encode_images(self, view1, view2):
        """
        Forward interface of correspondence prediction networks.
        """
        # Encode the two images --> Each feat output: BCHW features (batch_size, feature_dim, feature_height, feature_width)
        if "labels" in view1 and "labels" in view2:
            # use label-based feature caching
            all_images = torch.cat((view1["img"], view2["img"]), dim=0)
            all_labels = view1["labels"] + view2["labels"]

            all_feats = self._encode_and_cache_image_features(
                img=all_images,
                labels=all_labels,
                data_norm_type=view1["data_norm_type"],
            )

            return all_feats.chunk(2, dim=0)
        else:
            # use existing symmetric-based inference
            feat1_list, feat2_list = self._encode_symmetrized(view1, view2, view1["symmetrized"])

            return feat1_list[-1], feat2_list[-1]

    def encoder_feats_to_output(self, encoder_feat1: torch.Tensor, encoder_feat2: torch.Tensor):
        """
        Encoded features to final outputs.
        """
        
        # Pass the features through the info_sharing
        info_sharing_input = MultiViewTransformerInput(features=[encoder_feat1, encoder_feat2])

        final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.info_sharing(
            info_sharing_input
        )

        info_sharing_outputs = {
            "1": [
                encoder_feat1.float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[0].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[0].float().contiguous(),
                final_info_sharing_multi_view_feat.features[0].float().contiguous(),
            ],
            "2": [
                encoder_feat2.float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[1].float().contiguous(),
                final_info_sharing_multi_view_feat.features[1].float().contiguous(),
            ],
        }

        result = UFMOutputInterface()

        # The prediction need precision, so we disable any autocasting here
        with torch.autocast("cuda", torch.float32):
            # run the collected info_sharing features through the prediction heads
            B, _, Hp, Wp = encoder_feat1.shape
            shape1 = (Hp * self.encoder.patch_size, Wp * self.encoder.patch_size)
            head_output1 = self._downstream_head(1, info_sharing_outputs, shape1)

            if "flow" in head_output1:
                # output is flow only
                result.flow = UFMFlowFieldOutput(flow_output=head_output1["flow"].value)

            if "flow_cov" in head_output1:
                result.flow.flow_covariance = head_output1["flow_cov"].covariance
                result.flow.flow_covariance_inv = head_output1["flow_cov"].inv_covariance
                result.flow.flow_covariance_log_det = head_output1["flow_cov"].log_det

            if "non_occluded_mask" in head_output1:
                result.covisibility = UFMMaskFieldOutput(
                    mask=head_output1["non_occluded_mask"].mask,
                    logits=head_output1["non_occluded_mask"].logits,
                )

        return result


    def forward(self, view1, view2) -> UFMOutputInterface:
        """
        Forward interface of correspondence prediction networks.

        Args:
        - view1 (Dict[str, Any]): Input view 1
          - img (torch.Tensor): BCHW image tensor normalized according to encoder's data_norm_type
          - instance (List[int]): List of instance indices, or id of the input image
          - data_norm_type (str): Data normalization type, see uniception.models.encoders.IMAGE_NORMALIZATION_DICT
        - view2 (Dict[str, Any]): Input view 2
          - (same structure as view1)
        Returns:
        - Dict[str, Any]: Output results
          - flow [Required] (Dict[str, torch.Tensor]): Flow output
            - [Required] flow_output (torch.Tensor): Flow output tensor, BCHW
            - [Optional] flow_covariance
            - [Optional] flow_covariance_inv
            - [Optional] flow_covariance_log_det
          - covisibility [Optional] (Dict[str, torch.Tensor]): Covisibility output
            - [Optional] mask
            - [Optional] logits
        """

        feat1, feat2 = self.encode_images(view1, view2)
        return self.encoder_feats_to_output(feat1, feat2)

    def inference(self, input_A, input_B) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MAC-VO inference interface.

        Inputs:
            - input_A: the first images
            - input_B: the second images

        Outputs:
            - est_flow: the estimated flow
            - est_cov: the estimated covariance
        """

        # FIXME: We make a assumption here that images are BCHW, normalized to [0, 1]
        views = [
            {
                "img": input_A,
                "data_norm_type": "dummy",
                "symmetrized": False
            },
            {
                "img": input_B,
                "data_norm_type": "dummy",
                "symmetrized": False
            }
        ]

        with torch.inference_mode():
            output = model(views[0], views[1])

        return output.flow.flow_output, output.flow.flow_covariance[..., :2]

    def _downstream_head(self, head_num, decout, img_shape):
        "Run the respective prediction heads"
        # if self.info_sharing_and_head_structure == "dual+single":

        head = getattr(self, f"head{head_num}")
        if self.head_type == "linear":
            head_input = PredictionHeadInput(last_feature=decout[f"{head_num}"])
        elif self.head_type in ["dpt", "moge_conv"]:
            head_input = PredictionHeadLayeredInput(list_features=decout[f"{head_num}"], target_output_shape=img_shape)

        return head(head_input)

    def get_parameter_groups(self) -> Dict[str, torch.nn.ParameterList]:
        """
        Get parameter groups for optimizer. This methods guides the optimizer
        to apply correct learning rate to different parts of the model.

        Returns:
        - Dict[str, torch.nn.ParameterList]: Parameter groups for optimizer
        """

        params_dict = {
            "encoder": torch.nn.ParameterList(self.encoder.parameters()),
            "info_sharing": torch.nn.ParameterList(self.info_sharing.parameters()),
            "output_head": torch.nn.ParameterList(self.head1.parameters()),
        }

        if self.cnn_augment:
            params_dict["cnn_encoder"] = torch.nn.ParameterList(
                list(self.cnn_encoder.parameters()) + list(self.cnn_feat_proj.parameters())
            )

        return params_dict


class UniFlowMatchConfidence(UniFlowMatch, PyTorchModelHubMixin):
    """
    UniFlowMatch model with uncertainty estimation.
    """

    def __init__(
        self,
        # Encoder configurations
        encoder_str: str,
        encoder_kwargs: Dict[str, Any],
        # Info sharing & output head structure configurations
        info_sharing_and_head_structure: str = "dual+single",  # only dual+single is supported
        # Information sharing configurations
        info_sharing_str: str = "global_attention",
        info_sharing_kwargs: Dict[str, Any] = {},
        # Prediction Heads & Adaptors
        head_type: str = "dpt",
        feature_head_kwargs: Dict[str, Any] = {},
        adaptors_kwargs: Dict[str, Any] = {},
        # Uncertainty Heads & Adaptors
        detach_uncertainty_head: bool = True,
        uncertainty_head_type: str = "dpt",
        uncertainty_head_kwargs: Dict[str, Any] = {},
        uncertainty_adaptors_kwargs: Dict[str, Any] = {},
        # Load Pretrained Weights
        pretrained_backbone_checkpoint_path: Optional[str] = None,
        pretrained_checkpoint_path: Optional[str] = None,
        # Inference Settings
        inference_resolution: Optional[Tuple[int, int]] = (560, 420),  # WH
        *args,
        **kwargs,
    ):
        UniFlowMatch.__init__(
            self,
            encoder_str=encoder_str,
            encoder_kwargs=encoder_kwargs,
            info_sharing_and_head_structure=info_sharing_and_head_structure,
            info_sharing_str=info_sharing_str,
            info_sharing_kwargs=info_sharing_kwargs,
            head_type=head_type,
            feature_head_kwargs=feature_head_kwargs,
            adaptors_kwargs=adaptors_kwargs,
            pretrained_checkpoint_path=pretrained_backbone_checkpoint_path,
            inference_resolution=inference_resolution,
            *args,
            **kwargs,
        )

        PyTorchModelHubMixin.__init__(self)

        # initialize uncertainty heads
        assert uncertainty_head_type == "dpt", "Only DPT is supported for uncertainty head now"

        self.uncertainty_head = self._initialize_prediction_heads(
            uncertainty_head_type, uncertainty_head_kwargs, uncertainty_adaptors_kwargs
        )
        self.uncertainty_adaptors = self._initialize_adaptors(uncertainty_adaptors_kwargs)

        assert pretrained_checkpoint_path is None, "Pretrained weights are not supported for now"

        self.detach_uncertainty_head = detach_uncertainty_head

    def forward(self, view1, view2) -> UFMOutputInterface:
        """
        Forward interface of correspondence prediction networks.

        Args:
        - view1 (Dict[str, Any]): Input view 1
          - img (torch.Tensor): BCHW image tensor normalized according to encoder's data_norm_type
          - instance (List[int]): List of instance indices, or id of the input image
          - data_norm_type (str): Data normalization type, see uniception.models.encoders.IMAGE_NORMALIZATION_DICT
        - view2 (Dict[str, Any]): Input view 2
          - (same structure as view1)
        Returns:
        - Dict[str, Any]: Output results
          - flow [Required] (Dict[str, torch.Tensor]): Flow output
            - [Required] flow_output (torch.Tensor): Flow output tensor, BCHW
            - [Optional] flow_covariance
            - [Optional] flow_covariance_inv
            - [Optional] flow_covariance_log_det
          - covisibility [Optional] (Dict[str, torch.Tensor]): Covisibiltiy output
            - [Optional] mask
            - [Optional] logits
        """

        # Get input shapes
        _, _, height1, width1 = view1["img"].shape
        _, _, height2, width2 = view2["img"].shape
        shape1 = (int(height1), int(width1))
        shape2 = (int(height2), int(width2))

        # Encode the two images --> Each feat output: BCHW features (batch_size, feature_dim, feature_height, feature_width)
        feat1_list, feat2_list = self._encode_symmetrized(view1, view2, view1["symmetrized"])

        # Pass the features through the info_sharing
        info_sharing_input = MultiViewTransformerInput(features=[feat1_list[-1], feat2_list[-1]])

        final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.info_sharing(
            info_sharing_input
        )

        info_sharing_outputs = {
            "1": [
                feat1_list[-1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[0].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[0].float().contiguous(),
                final_info_sharing_multi_view_feat.features[0].float().contiguous(),
            ],
            "2": [
                feat2_list[-1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[1].float().contiguous(),
                final_info_sharing_multi_view_feat.features[1].float().contiguous(),
            ],
        }

        info_sharing_outputs_detached = {
            "1": [
                feat1_list[-1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[0].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[0].detach().float().contiguous(),
                final_info_sharing_multi_view_feat.features[0].detach().float().contiguous(),
            ],
            "2": [
                feat2_list[-1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[1].detach().float().contiguous(),
                final_info_sharing_multi_view_feat.features[1].detach().float().contiguous(),
            ],
        }

        result = UFMOutputInterface()

        # The prediction need precision, so we disable any autocasting here
        with torch.autocast("cuda", torch.float32):
            # run the collected info_sharing features through the prediction heads
            head_output1 = self._downstream_head(1, info_sharing_outputs, shape1)
            head_output_uncertainty = self._downstream_head(
                "uncertainty",
                info_sharing_outputs_detached if self.detach_uncertainty_head else info_sharing_outputs,
                shape1,
            )

            result.flow = UFMFlowFieldOutput(
                flow_output=head_output1["flow"].value,
            )

            if "flow_cov" in head_output_uncertainty:
                result.flow.flow_covariance = head_output_uncertainty["flow_cov"].covariance
                result.flow.flow_covariance_inv = head_output_uncertainty["flow_cov"].inv_covariance
                result.flow.flow_covariance_log_det = head_output_uncertainty["flow_cov"].log_det

            if "keypoint_confidence" in head_output_uncertainty:
                result.keypoint_confidence = head_output_uncertainty["keypoint_confidence"].value.squeeze(1)

            if "non_occluded_mask" in head_output_uncertainty:
                result.covisibility = UFMMaskFieldOutput(
                    mask=head_output_uncertainty["non_occluded_mask"].mask,
                    logits=head_output_uncertainty["non_occluded_mask"].logits,
                )

        return result

    def get_parameter_groups(self) -> Dict[str, torch.nn.ParameterList]:
        """
        Get parameter groups for optimizer. This methods guides the optimizer
        to apply correct learning rate to different parts of the model.

        Returns:
        - Dict[str, torch.nn.ParameterList]: Parameter groups for optimizer
        """

        return {
            "encoder": torch.nn.ParameterList(self.encoder.parameters()),
            "info_sharing": torch.nn.ParameterList(self.info_sharing.parameters()),
            "output_head": torch.nn.ParameterList(self.head1.parameters()),
            "uncertainty_head": torch.nn.ParameterList(self.uncertainty_head.parameters()),
        }

    def _downstream_head(self, head_num, decout, img_shape):
        "Run the respective prediction heads"
        # if self.info_sharing_and_head_structure == "dual+single":

        head = getattr(self, f"head{head_num}") if head_num != "uncertainty" else self.uncertainty_head

        head_num = head_num if head_num != "uncertainty" else 1  # uncertainty head is always from branch 1

        if self.head_type == "linear":
            head_input = PredictionHeadInput(last_feature=decout[f"{head_num}"])
        elif self.head_type in ["dpt", "moge_conv"]:
            head_input = PredictionHeadLayeredInput(list_features=decout[f"{head_num}"], target_output_shape=img_shape)

        return head(head_input)


class UniFlowMatchClassificationRefinement(UniFlowMatch, PyTorchModelHubMixin):
    """
    The variant of UniFlowMatch with local classification for refinement.
    """

    def __init__(
        self,
        # Encoder configurations
        encoder_str: str,
        encoder_kwargs: Dict[str, Any],
        # Info sharing & output head structure configurations
        info_sharing_and_head_structure: str = "dual+single",  # only dual+single is supported
        # Information sharing configurations
        info_sharing_str: str = "global_attention",
        info_sharing_kwargs: Dict[str, Any] = {},
        # Prediction Heads & Adaptors
        head_type: str = "dpt",
        feature_head_kwargs: Dict[str, Any] = {},
        adaptors_kwargs: Dict[str, Any] = {},
        # Uncertainty Heads & Adaptors
        detach_uncertainty_head: bool = True,
        uncertainty_head_type: str = "dpt",
        uncertainty_head_kwargs: Dict[str, Any] = {},
        uncertainty_adaptors_kwargs: Dict[str, Any] = {},
        # Classification Heads & Adaptors
        temperature: float = 4,
        use_unet_feature: bool = False,
        classification_head_type: str = "patch_mlp",
        classification_head_kwargs: Dict[str, Any] = {},
        feature_combine_method: str = "conv",
        # Refinement Range
        refinement_range: int = 5,
        # Load Pretrained Weights
        pretrained_backbone_checkpoint_path: Optional[str] = None,
        pretrained_checkpoint_path: Optional[str] = None,
        # Inference Settings
        inference_resolution: Optional[Tuple[int, int]] = (560, 420),  # WH
        *args,
        **kwargs,
    ):
        UniFlowMatch.__init__(
            self,
            encoder_str=encoder_str,
            encoder_kwargs=encoder_kwargs,
            info_sharing_and_head_structure=info_sharing_and_head_structure,
            info_sharing_str=info_sharing_str,
            info_sharing_kwargs=info_sharing_kwargs,
            head_type=head_type,
            feature_head_kwargs=feature_head_kwargs,
            adaptors_kwargs=adaptors_kwargs,
            pretrained_checkpoint_path=pretrained_backbone_checkpoint_path,
            inference_resolution=inference_resolution,
            *args,
            **kwargs,
        )

        PyTorchModelHubMixin.__init__(self)

        # initialize uncertainty heads
        assert classification_head_type == "patch_mlp", "Only DPT is supported for uncertainty head now"
        self.classification_head_type = classification_head_type

        self.classification_head = self._initialize_classification_head(classification_head_kwargs)

        self.refinement_range = refinement_range
        self.temperature = temperature

        assert pretrained_checkpoint_path is None, "Pretrained weights are not supported for now"

        self.use_unet_feature = use_unet_feature

        self.feature_combine_method = feature_combine_method

        # Unet experiment
        if self.use_unet_feature:
            self.unet_feature = UNet(in_channels=3, out_channels=16, features=[64, 128, 256, 512])

            self.conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

            if self.feature_combine_method == "conv":
                self.conv2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
            elif self.feature_combine_method == "modulate":
                self.conv2 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)

        default_attention_bias = torch.zeros(self.refinement_range * self.refinement_range)
        self.classification_bias = nn.Parameter(default_attention_bias)

        # initialize uncertainty heads
        if len(uncertainty_head_kwargs) > 0:
            assert uncertainty_head_type == "dpt", "Only DPT is supported for uncertainty head now"

            self.uncertainty_head = self._initialize_prediction_heads(
                uncertainty_head_type, uncertainty_head_kwargs, uncertainty_adaptors_kwargs
            )
            self.uncertainty_adaptors = self._initialize_adaptors(uncertainty_adaptors_kwargs)

            assert pretrained_checkpoint_path is None, "Pretrained weights are not supported for now"

            self.detach_uncertainty_head = detach_uncertainty_head

    def forward(self, view1, view2) -> UFMOutputInterface:
        """
        Forward interface of correspondence prediction networks.

        Args:
        - view1 (Dict[str, Any]): Input view 1
          - img (torch.Tensor): BCHW image tensor normalized according to encoder's data_norm_type
          - instance (List[int]): List of instance indices, or id of the input image
          - data_norm_type (str): Data normalization type, see uniception.models.encoders.IMAGE_NORMALIZATION_DICT
        - view2 (Dict[str, Any]): Input view 2
          - (same structure as view1)
        Returns:
        - Dict[str, Any]: Output results
          - flow [Required] (Dict[str, torch.Tensor]): Flow output
            - [Required] flow_output (torch.Tensor): Flow output tensor, BCHW
            - [Optional] flow_covariance
            - [Optional] flow_covariance_inv
            - [Optional] flow_covariance_log_det
          - covisibility [Optional] (Dict[str, torch.Tensor]): Covisibility output
            - [Optional] mask
            - [Optional] logits
          - classification [Optional]: Probability and targets of the classification head
        """

        # Get input shapes
        _, _, height1, width1 = view1["img"].shape
        _, _, height2, width2 = view2["img"].shape
        shape1 = (int(height1), int(width1))
        shape2 = (int(height2), int(width2))

        # Encode the two images --> Each feat output: BCHW features (batch_size, feature_dim, feature_height, feature_width)
        feat1_list, feat2_list = self._encode_symmetrized(view1, view2, view1["symmetrized"])

        # Pass the features through the info_sharing
        info_sharing_input = MultiViewTransformerInput(features=[feat1_list[-1], feat2_list[-1]])

        final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.info_sharing(
            info_sharing_input
        )

        info_sharing_outputs = {
            "1": [
                feat1_list[-1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[0].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[0].float().contiguous(),
                final_info_sharing_multi_view_feat.features[0].float().contiguous(),
            ],
            "2": [
                feat2_list[-1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[1].float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[1].float().contiguous(),
                final_info_sharing_multi_view_feat.features[1].float().contiguous(),
            ],
        }

        info_sharing_outputs_detached = {
            "1": [
                feat1_list[-1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[0].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[0].detach().float().contiguous(),
                final_info_sharing_multi_view_feat.features[0].detach().float().contiguous(),
            ],
            "2": [
                feat2_list[-1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[0].features[1].detach().float().contiguous(),
                intermediate_info_sharing_multi_view_feat[1].features[1].detach().float().contiguous(),
                final_info_sharing_multi_view_feat.features[1].detach().float().contiguous(),
            ],
        }

        # optionally inference for U-Net Features
        if self.use_unet_feature:
            unet_feat1 = self.unet_feature(view1["img"])
            unet_feat2 = self.unet_feature(view2["img"])

        result = UFMOutputInterface()
        # The prediction need precision, so we disable any autocasting here
        with torch.autocast("cuda", torch.float32):
            # run the collected info_sharing features through the prediction heads
            head_output1 = self._downstream_head(1, info_sharing_outputs, shape1)
            flow_prediction = head_output1["flow"].value

            if hasattr(self, "uncertainty_head"):
                # run the uncertainty head
                head_output_uncertainty = self._downstream_head(
                    "uncertainty",
                    info_sharing_outputs_detached if self.detach_uncertainty_head else info_sharing_outputs,
                    shape1,
                )

                if "flow_cov" in head_output_uncertainty:
                    result.flow.flow_covariance = head_output_uncertainty["flow_cov"].covariance
                    result.flow.flow_covariance_inv = head_output_uncertainty["flow_cov"].inv_covariance
                    result.flow.flow_covariance_log_det = head_output_uncertainty["flow_cov"].log_det

                if "keypoint_confidence" in head_output_uncertainty:
                    result.keypoint_confidence = head_output_uncertainty["keypoint_confidence"].value.squeeze(1)

                if "non_occluded_mask" in head_output_uncertainty:
                    result.covisibility = UFMMaskFieldOutput(
                        mask=head_output_uncertainty["non_occluded_mask"].mask,
                        logits=head_output_uncertainty["non_occluded_mask"].logits,
                    )

            # we run the classification head in the autocast environment bacause it is not regression
            if self.classification_head_type == "patch_mlp":
                # concatenate the last encoder feature with final info_sharing feature

                # use the first encoder feature, because it captures more low-level information, which is needed
                # for refinement of the regressed flow.
                classification_feat_1 = torch.cat(
                    [feat1_list[0].float().contiguous(), info_sharing_outputs["1"][-1]], dim=1
                )
                classification_feat_2 = torch.cat(
                    [feat2_list[0].float().contiguous(), info_sharing_outputs["2"][-1]], dim=1
                )

                classification_input = PredictionHeadInput(
                    torch.cat([classification_feat_1, classification_feat_2], dim=0)
                )

                classification_features = self.classification_head(classification_input).decoded_channels

                if self.use_unet_feature:

                    if self.feature_combine_method == "conv":
                        combined_features = torch.cat(
                            [classification_features, torch.cat([unet_feat1, unet_feat2], dim=0)], dim=1
                        )

                        combined_features = self.conv1(combined_features)
                        combined_features = nn.functional.relu(combined_features)
                        combined_features = self.conv2(combined_features)
                    elif self.feature_combine_method == "modulate":

                        combined_features = classification_features * torch.tanh(
                            torch.cat([unet_feat1, unet_feat2], dim=0)
                        )
                        combined_features = self.conv2(combined_features)

                    classification_features = combined_features

                classification_features0, classification_features1 = classification_features.chunk(2, dim=0)

            # refine the flow prediction with features from the classification head
            for i in range(1):
                residual, log_softmax_attention = self.classification_refinement(
                    flow_prediction, classification_features
                )
                flow_prediction = flow_prediction + residual

        # Fill in the result
        # WARNING: based on how the residual is computed, flow_prediction will have gradient cancelled by mathematics,
        # so there will be no supervision to the flow prediction at all. We need to use specialized loss function to
        # supervise the regression_flow_output.
        result.flow = UFMFlowFieldOutput(
            flow_output=flow_prediction,
        )

        result.classification_refinement = UFMClassificationRefinementOutput(
            regression_flow_output=flow_prediction,
            residual=residual,
            log_softmax=log_softmax_attention,
            feature_map_0=classification_features0,
            feature_map_1=classification_features1,
        )

        return result


    # @torch.compile()
    def classification_refinement(self, flow_prediction, classification_features) -> Dict[str, Any]:
        """
        Use correlation between self feature and features around a local patch of the initial flow prediction
        to refine the flow prediction.

        """

        classification_features1, classification_features2 = classification_features.chunk(2, dim=0)

        neighborhood_features, neighborhood_flow_residual = self.obtain_neighborhood_features(
            flow_estimation=flow_prediction, other_features=classification_features2, local_patch=self.refinement_range
        )

        residual, log_softmax_attention = self.compute_refinement_attention(
            classification_features1, neighborhood_features, neighborhood_flow_residual
        )

        return residual, log_softmax_attention

    def compute_refinement_attention(self, classification_features1, neighborhood_features, neighborhood_flow_residual):
        """
        Compute the attention for the refinement, with special processing
        to fit
        """

        B, C, H, W = classification_features1.shape
        P = self.refinement_range

        # reshape Q to B, H, W, 1, 1, C
        classification_features1 = classification_features1.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)

        # reshape K to B, H, W, 1, P^2, C
        assert neighborhood_features.shape[0] == B
        assert neighborhood_features.shape[1] == H
        assert neighborhood_features.shape[2] == W
        assert neighborhood_features.shape[3] == P
        assert neighborhood_features.shape[4] == P
        assert neighborhood_features.shape[5] == C

        neighborhood_features = neighborhood_features.reshape(B * H * W, P * P, C)

        # reshape V to B, H, W, 1, P^2, 2
        neighborhood_flow_residual = neighborhood_flow_residual.reshape(-1, P * P, 2)

        # compute the attention
        attention_score = (
            torch.matmul(classification_features1, neighborhood_features.permute(0, 2, 1)) / self.temperature
        )
        attention_score = attention_score + self.classification_bias

        attention = torch.nn.functional.softmax(attention_score, dim=-1)
        log_softmax_attention = torch.nn.functional.log_softmax(attention_score, dim=-1)

        # compute the weighted sum
        residual = torch.matmul(attention, neighborhood_flow_residual)

        # reshape the residual to B, H, W, 2, then B, 2, H, W
        residual = residual.reshape(B, H, W, 2).permute(0, 3, 1, 2)

        return residual, log_softmax_attention.reshape(B, H, W, P, P)

    def _downstream_head(self, head_num, decout, img_shape):
        "Run the respective prediction heads"
        # if self.info_sharing_and_head_structure == "dual+single":

        head = getattr(self, f"head{head_num}") if head_num != "uncertainty" else self.uncertainty_head

        head_num = head_num if head_num != "uncertainty" else 1  # uncertainty head is always from branch 1

        if self.head_type == "linear":
            head_input = PredictionHeadInput(last_feature=decout[f"{head_num}"])
        elif self.head_type in ["dpt", "moge_conv"]:
            head_input = PredictionHeadLayeredInput(list_features=decout[f"{head_num}"], target_output_shape=img_shape)

        return head(head_input)

    def obtain_neighborhood_features(
        self, flow_estimation: torch.Tensor, other_features: torch.Tensor, local_patch: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the other features according to flow estimation.
        """

        assert local_patch % 2 == 1, "local_patch should be odd number"

        P = local_patch
        R = (P - 1) // 2
        B, C, H, W = other_features.shape

        device = other_features.device

        # expected_output = torch.zeros(B, H, W, P, P, C, device=other_features.device, dtype=torch.float32)

        neighborhood_grid_ij: torch.Tensor

        i_local, j_local = torch.meshgrid(
            torch.arange(-R, R + 1, device=device), torch.arange(-R, R + 1, device=device), indexing="ij"
        )
        ij_local = torch.stack((i_local, j_local), dim=0)  # 2, P, P tensor

        # compute the indices of the fetch
        base_grid_xy = get_meshgrid_torch(W=W, H=H, device=device).permute(2, 0, 1).reshape(1, 2, H, W)

        target_coordinate_xy_float = flow_estimation + base_grid_xy
        target_coordinate_xy = target_coordinate_xy_float.view(B, 2, H, W, 1, 1)
        target_coordinate_ij = target_coordinate_xy[:, [1, 0], ...]

        # compute the neighborhood grid
        neighborhood_grid_ij = target_coordinate_ij + ij_local.view(1, 2, 1, 1, P, P)

        grid_for_sample = neighborhood_grid_ij[:, [1, 0], ...].permute(0, 2, 3, 4, 5, 1).reshape(B, H, W * P * P, 2)
        grid_for_sample = (grid_for_sample + 0.5) / torch.tensor([W, H], device=device).view(1, 1, 1, 2)
        grid_for_sample = grid_for_sample * 2 - 1

        expected_output = torch.nn.functional.grid_sample(
            other_features, grid=grid_for_sample, mode="bicubic", padding_mode="zeros", align_corners=False
        ).view(B, C, H, W, P, P)

        # transform BCHWPP to BHWPPC
        expected_output = expected_output.permute(0, 2, 3, 4, 5, 1)

        neighborhood_grid_xy_residual = ij_local[[1, 0], ...].view(1, 2, 1, 1, P, P).to(device).float()
        neighborhood_grid_xy_residual = neighborhood_grid_xy_residual.permute(0, 2, 3, 4, 5, 1).float()

        return expected_output, neighborhood_grid_xy_residual

    def _initialize_classification_head(self, classification_head_kwargs: Dict[str, Any]):
        """
        Initialize classification head

        Args:
        - classification_head_kwargs (Dict[str, Any]): Classification head configurations

        Returns:
        - nn.Module: Classification head
        """

        if self.classification_head_type == "patch_mlp":
            return MLPFeature(**classification_head_kwargs)
        else:
            raise ValueError(f"Classification head type {self.classification_head_type} not supported.")

    def get_parameter_groups(self) -> Dict[str, torch.nn.ParameterList]:
        """
        Get parameter groups for optimizer. This methods guides the optimizer
        to apply correct learning rate to different parts of the model.

        Returns:
        - Dict[str, torch.nn.ParameterList]: Parameter groups for optimizer
        """

        if self.use_unet_feature:
            params_dict = {
                "encoder": torch.nn.ParameterList(self.encoder.parameters()),
                "info_sharing": torch.nn.ParameterList(self.info_sharing.parameters()),
                "output_head": torch.nn.ParameterList(self.head1.parameters()),
                "classification_head": torch.nn.ParameterList(self.classification_head.parameters()),
                "unet_feature": torch.nn.ParameterList(
                    list(self.unet_feature.parameters())
                    + list(self.conv1.parameters())
                    + list(self.conv2.parameters())
                    + [self.classification_bias]
                ),
            }
        else:
            params_dict = {
                "encoder": torch.nn.ParameterList(self.encoder.parameters()),
                "info_sharing": torch.nn.ParameterList(self.info_sharing.parameters()),
                "output_head": torch.nn.ParameterList(self.head1.parameters()),
                "classification_head": torch.nn.ParameterList(self.classification_head.parameters()),
            }

        if hasattr(self, "uncertainty_head"):
            params_dict["uncertainty_head"] = torch.nn.ParameterList(self.uncertainty_head.parameters())

        return params_dict


if __name__ == "__main__":
    import cv2
    import flow_vis
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from tqdm import tqdm

    from uniflowmatch.utils.geometry import get_meshgrid_torch
    from uniflowmatch.utils.viz import warp_image_with_flow

    USE_REFINEMENT_MODEL = False

    model = UniFlowMatch.from_pretrained("infinity1096/UFM-Robotics-V0", token="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", inference_resolution=[(480, 320)])
    model.to("cuda")
    model.eval()

    img_L = torch.randn(2, 3, 320, 480).cuda()
    img_R = torch.randn(2, 3, 320, 480).cuda()

    for t in tqdm(range(1000)):
        views = [
            {
                "img": img_L,
                "labels": [f"{t}_L", f"{t+1}_L"],
                "data_norm_type": "radio",
                # "symmetrized": False
            },
            {
                "img": img_R,
                "labels": [f"{t}_R", f"{t+1}_R"],
                "data_norm_type": "radio",
                # "symmetrized": False
            }
        ]

        with torch.inference_mode():
            output = model(views[0], views[1])