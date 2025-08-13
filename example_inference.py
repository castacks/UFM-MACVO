"""
Example inference script for UFM (UniFlowMatch) models.
This script demonstrates how to load a pre-trained UFM model, predict correspondences
between two images, and visualize the results including flow output and covisibility mask.

Usage:
    python example_inference_enhanced.py --source examples/image_pairs/fire_academy_0.png --target examples/image_pairs/fire_academy_1.png
    python example_inference_enhanced.py --model refine --source img1.jpg --target img2.jpg --output results.png
"""

import argparse

import cv2
import flow_vis
import matplotlib.pyplot as plt
import numpy as np
import torch

from uniflowmatch.models.ufm import UniFlowMatch, UniFlowMatchClassificationRefinement, UniFlowMatchConfidence
from uniflowmatch.utils.viz import warp_image_with_flow


def load_image(image_path):
    """Load and preprocess an image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def predict_correspondences(model, source_image, target_image):
    """Predict correspondences between source and target images."""
    with torch.no_grad():
        result = model.predict_correspondences_batched(
            source_image=torch.from_numpy(source_image),
            target_image=torch.from_numpy(target_image),
        )

        flow_output = result.flow.flow_output[0].cpu().numpy()
        covisibility = result.covisibility.mask[0].cpu().numpy()

        if hasattr(result.flow, 'flow_covariance'):
            covariance = result.flow.flow_covariance[0].cpu().numpy()
        else:
            covariance = None

    return flow_output, covisibility, covariance


def visualize_results(source_image, target_image, flow_output, covisibility, covariance=None, output_path="ufm_output.png"):
    """Create and save visualization of results."""
    fig, axs = plt.subplots(2 if covariance is None else 3, 3, figsize=(15, 10))

    # Top row: Input images and warped result
    axs[0, 0].imshow(source_image)
    axs[0, 0].set_title("Source Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(target_image)
    axs[0, 1].set_title("Target Image")
    axs[0, 1].axis("off")

    # Warp the image using flow
    warped_image = warp_image_with_flow(source_image, None, target_image, flow_output.transpose(1, 2, 0))
    warped_image = covisibility[..., None] * warped_image + (1 - covisibility[..., None]) * 255 * np.ones_like(
        warped_image
    )
    warped_image = np.clip(warped_image / 255.0, 0, 1)

    axs[0, 2].imshow(warped_image)
    axs[0, 2].set_title("Warped Source Image")
    axs[0, 2].axis("off")

    # Bottom row: Flow and covisibility visualizations
    flow_vis_image = flow_vis.flow_to_color(flow_output.transpose(1, 2, 0))
    axs[1, 0].imshow(flow_vis_image)
    axs[1, 0].set_title("Flow Visualization (Valid at Covisible Pixels)")
    axs[1, 0].axis("off")

    # Covisibility mask (thresholded)
    axs[1, 1].imshow(covisibility > 0.5, cmap="gray", vmin=0, vmax=1)
    axs[1, 1].set_title("Covisibility Mask (>0.5)")
    axs[1, 1].axis("off")

    # Covisibility mask (continuous)
    heatmap = axs[1, 2].imshow(covisibility, cmap="viridis", vmin=0, vmax=1)
    axs[1, 2].set_title("Covisibility Confidence")
    axs[1, 2].axis("off")
    plt.colorbar(heatmap, ax=axs[1, 2], shrink=0.6)

    # Visualize covariance if it is not none
    if covariance is not None:
        axs[2, 0].imshow(np.log(np.sqrt(covariance[0] + 1e-4)), cmap="viridis")
        axs[2, 0].set_title("Covariance XX")
        axs[2, 0].axis("off")

        axs[2, 1].imshow(np.log(np.sqrt(covariance[1] + 1e-4)), cmap="viridis")
        axs[2, 1].set_title("Covariance YY")
        axs[2, 1].axis("off")

        axs[2, 2].imshow(covariance[2], cmap="viridis")
        axs[2, 2].set_title("Covariance XY")
        axs[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="UFM inference example")
    parser.add_argument(
        "--source", "-s", default="examples/image_pairs/fire_academy_0.png", help="Path to source image"
    )
    parser.add_argument(
        "--target", "-t", default="examples/image_pairs/fire_academy_1.png", help="Path to target image"
    )
    parser.add_argument("--output", "-o", default="ufm_output.png", help="Output visualization path")
    parser.add_argument("--show", action="store_true", help="Display the visualization")

    args = parser.parse_args()

    # Load model
    model = UniFlowMatch.from_pretrained("infinity1096/UFM-Robotics-V0", token="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", inference_resolution=[(480, 320)])

    model.eval()
    print("Model loaded successfully!")

    # Load and prepare images
    print(f"Loading images: {args.source}, {args.target}")
    source_image = load_image(args.source)
    target_image = load_image(args.target)

    print(f"Image shapes: {source_image.shape}, {target_image.shape}")

    # Predict correspondences
    print("Running inference...")
    flow_output, covisibility, covariance = predict_correspondences(model, source_image, target_image)

    # Visualize results
    fig = visualize_results(source_image, target_image, flow_output, covisibility, covariance, args.output)

    if args.show:
        plt.show()

    print("Inference completed!")


if __name__ == "__main__":
    main()