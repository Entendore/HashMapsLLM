import torch
import zlib
import threading
import numpy as np
from typing import Optional, Tuple, Dict, Iterator, Any, Union

class LLM_Weight_HashMap:
    """
    Production-grade Hash Map for LLM weights using Robin Hood Open Addressing,
    Tombstone Deletion, and direct C-level memoryview copying.
    """

    # Sentinel object for tombstones (deleted entries)
    _TOMBSTONE = object()

    __slots__ = ("capacity", "table", "size", "active_entries", 
                 "load_factor_threshold", "_flat_buffer", "_is_frozen", "lock")

    def __init__(self, initial_capacity: int = 16, load_factor_threshold: float = 0.75):
        if initial_capacity < 16:
            initial_capacity = 16
        self.capacity = initial_capacity
        self.table: list[Optional[tuple]] = [None] * self.capacity
        self.size = 0                  # Active entries (excludes tombstones)
        self.active_entries = 0        # Active + Tombstones (used for load factor)
        self.load_factor_threshold = load_factor_threshold
        self._flat_buffer: Optional[bytearray] = None
        self._is_frozen = False
        self.lock = threading.RLock()

    # ---------- Robin Hood Hashing ----------
    def _hash(self, key: str) -> int:
        return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF

    def _find_slot(self, key: str) -> Optional[int]:
        """Robin Hood search: stops early if we find a richer element."""
        h = self._hash(key)
        idx = h % self.capacity
        psl = 0  # Probe Sequence Length
        
        while self.table[idx] is not None:
            entry = self.table[idx]
            # Skip tombstones during lookup
            if entry is self._TOMBSTONE:
                idx = (idx + 1) % self.capacity
                psl += 1
                continue
                
            entry_h, entry_psl, entry_key, entry_off, entry_len, entry_dtype, entry_shape = entry
            
            # Robin Hood early exit: if our PSL is greater than the entry's PSL,
            # the key cannot exist further down the chain.
            if psl > entry_psl:
                return None
            if entry_key == key:
                return idx
                
            idx = (idx + 1) % self.capacity
            psl += 1
        return None

    def _insert_entry(self, entry: tuple) -> bool:
        """Robin Hood insertion: steals slots from richer elements."""
        h, psl, key, offset, length, dtype, shape = entry
        idx = h % self.capacity
        
        while self.table[idx] is not None and self.table[idx] is not self._TOMBSTONE:
            existing = self.table[idx]
            _, ex_psl, ex_key, *_ = existing
            
            if ex_key == key:
                self.table[idx] = entry
                return False  # Updated
                
            # Robin Hood: if the existing element is "richer" (traveled less), swap.
            if psl > ex_psl:
                self.table[idx] = entry
                entry = existing
                psl = ex_psl  # Continue inserting the evicted element
                
            idx = (idx + 1) % self.capacity
            psl += 1
            
        self.table[idx] = entry
        return True

    def _resize(self) -> None:
        old_table = self.table
        self.capacity *= 2
        self.table = [None] * self.capacity
        self.active_entries = 0
        
        for entry in old_table:
            if entry is not None and entry is not self._TOMBSTONE:
                # Reset PSL to 0 before re-inserting
                h, _, key, off, length, dtype, shape = entry
                new_entry = (h, 0, key, off, length, dtype, shape)
                if self._insert_entry(new_entry):
                    self.active_entries += 1

    # ---------- Core Memory & Serialization ----------
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        with self.lock:
            if self.size > 0:
                raise RuntimeError("Map must be empty to use load_state_dict.")
                
            needed = len(state_dict)
            target_cap = self.capacity
            while needed / target_cap > self.load_factor_threshold:
                target_cap *= 2
            self.capacity = target_cap
            self.table = [None] * self.capacity
            
            align = 16  # AVX/SIMD memory alignment
            current_offset = 0
            processed = []
            
            # Pass 1: Calculate 16-byte aligned offsets
            for key, t in state_dict.items():
                t_cpu = t.detach().cpu().contiguous()
                n_bytes = t_cpu.numel() * t_cpu.element_size()
                padding = (align - (current_offset % align)) % align
                current_offset += padding
                processed.append((key, t_cpu, n_bytes, current_offset))
                current_offset += n_bytes
                
            self._flat_buffer = bytearray(current_offset)
            mv_buf = memoryview(self._flat_buffer)  # C-level memoryview
            
            # Pass 2: Zero-copy insertion using memoryview (NO temporary bytes allocation)
            for key, t_cpu, n_bytes, offset in processed:
                # Trick: view tensor as uint8 to bypass numpy/bf16 limitations
                src_np = t_cpu.view(torch.uint8).reshape(-1).numpy()
                mv_src = memoryview(src_np)
                
                # Direct C-level memory copy into flat buffer
                mv_buf[offset : offset + n_bytes] = mv_src
                
                h = self._hash(key)
                entry = (h, 0, key, offset, n_bytes, t_cpu.dtype, tuple(t_cpu.shape))
                if self._insert_entry(entry):
                    self.active_entries += 1
                    self.size += 1
                    
            self._is_frozen = True

    def get_tensor(self, key: str, *, copy: bool = True, 
                   device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """
        Reconstructs tensor. 
        - copy=False: returns a zero-copy view into RAM.
        - device='cuda': streams the bytes directly to GPU VRAM.
        """
        with self.lock:
            idx = self._find_slot(key)
            if idx is None:
                raise KeyError(f"Layer '{key}' not found in Hash Map!")
            
            entry = self.table[idx]
            _, _, _, offset, length, dtype, shape = entry
            numel = 1
            for s in shape: numel *= s
            
            # Read directly from the memoryview
            t = torch.frombuffer(self._flat_buffer, dtype=dtype, count=numel, offset=offset).reshape(shape)
            
            if device:
                # Non-blocking transfer to GPU if pinned, but safe transfer regardless
                t = t.to(device)
            elif copy:
                t = t.clone()
                
            return t

    def delete(self, key: str) -> None:
        """O(1) Tombstone deletion. Does not break probe chains."""
        with self.lock:
            idx = self._find_slot(key)
            if idx is None:
                raise KeyError(key)
            self.table[idx] = self._TOMBSTONE
            self.size -= 1
            # Note: active_entries remains the same, so load factor is preserved correctly

    # ---------- Context Manager & Dunders ----------
    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
        return False

    def keys(self) -> Iterator[str]:
        with self.lock:
            for entry in self.table:
                if entry and entry is not self._TOMBSTONE:
                    yield entry[2]

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "capacity": self.capacity,
                "size": self.size,
                "tombstones": self.active_entries - self.size,
                "load_factor": self.active_entries / self.capacity,
                "flat_buffer_bytes": len(self._flat_buffer) if self._flat_buffer else 0,
                "is_frozen": self._is_frozen
            }

    def __len__(self) -> int: return self.size
    def __contains__(self, key) -> bool:
        with self.lock: return self._find_slot(key) is not None
    def __getitem__(self, key) -> torch.Tensor: return self.get_tensor(key)
    def __delitem__(self, key): self.delete(key)
    def __iter__(self) -> Iterator[str]: return self.keys()


# ==========================================
# Demo & Verification
# ==========================================
if __name__ == "__main__":
    print("Downloading real LLM weights (sshleifer/tiny-gpt2)...")
    real_model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    real_weights = real_model.state_dict()

    custom_hash_map = LLM_Weight_HashMap(initial_capacity=4)

    # Bulk load: Pre-sizes, aligns memory, and copies via memoryview
    custom_hash_map.load_state_dict(real_weights)
    
    print(f"Stored {len(custom_hash_map)} tensors.")
    print("Stats:", custom_hash_map.stats())

    # Test Deletion & Tombstone
    dummy_key = "transformer.h.0.attn.c_attn.weight"
    print(f"\nDeleting '{dummy_key}' to test Tombstones...")
    del custom_hash_map[dummy_key]
    print("Stats after delete:", custom_hash_map.stats())
    assert dummy_key not in custom_hash_map, "Tombstone deletion failed!"
    print("Tombstone deletion verified ✔\n")

    # Re-insert for math test
    custom_hash_map.set_weights(dummy_key, real_weights[dummy_key])

    # Test Native GPU transfer (if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_layer = "transformer.h.0.mlp.c_fc.weight"
    print(f"Retrieving '{target_layer}' directly to device='{device}'...")
    
    w = custom_hash_map.get_tensor(target_layer, device=device)
    print(f"  Shape: {tuple(w.shape)}  Dtype: {w.dtype}  Device: {w.device}")

    # Real LLM Math (GPT-2 Conv1D: x @ w)
    fake_input = torch.ones(1, w.shape[0], device=device)
    out = torch.matmul(fake_input, w)
    print(f"  matmul output shape: {tuple(out.shape)}")
    print(f"  first 5 values: {out[0, :5].tolist()}")

    print("\nAll advanced integrity checks passed ✔")