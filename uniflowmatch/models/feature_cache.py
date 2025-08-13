import torch
from collections import deque

class PreallocatedTensorLRU:
    def __init__(self, max_items: int, feature_shape: tuple, device=None, dtype=torch.float32):
        """
        Args:
            max_items (int): Maximum number of tensors to store.
            feature_shape (tuple): Shape of each individual tensor (e.g., (128, 256)).
            device: torch device (e.g., "cuda" or "cpu").
            dtype: tensor dtype.
        """
        self.max_items = max_items
        self.feature_shape = feature_shape
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Allocate a big tensor [max_items, *feature_shape]
        self.storage = torch.empty((max_items, *feature_shape), device=self.device, dtype=self.dtype)

        # Mapping key -> index in storage
        self.key_to_idx = {}
        self.idx_to_key = {}
        self.free_indices = deque(range(max_items))  # available slots
        self.lru_order = deque()  # keys in LRU order

    def to(self, device):
        """Move all tensors to the specified device."""
        self.storage = self.storage.to(device)
        self.device = device

    def get(self, key):
        """Retrieve tensor (view, not a copy) and mark as recently used."""
        if key not in self.key_to_idx:
            return None
        self._mark_used(key)
        return self.storage[self.key_to_idx[key]]

    def put(self, key, tensor: torch.Tensor):
        """Store tensor in preallocated storage, evicting LRU if needed."""
        if tensor.shape != self.feature_shape:
            raise ValueError(f"Expected shape {self.feature_shape}, got {tensor.shape}")

        if key in self.key_to_idx:
            idx = self.key_to_idx[key]
            self.storage[idx].copy_(tensor)
            self._mark_used(key)
            return

        # Evict if full
        if not self.free_indices:
            lru_key = self.lru_order.popleft()
            idx = self.key_to_idx.pop(lru_key)
            del self.idx_to_key[idx]
            self.free_indices.append(idx)

        idx = self.free_indices.popleft()
        self.storage[idx].copy_(tensor)
        self.key_to_idx[key] = idx
        self.idx_to_key[idx] = key
        self.lru_order.append(key)

    def _mark_used(self, key):
        """Move key to the back of LRU order."""
        self.lru_order.remove(key)
        self.lru_order.append(key)

    def __contains__(self, key):
        return key in self.key_to_idx

    def __len__(self):
        return len(self.key_to_idx)
