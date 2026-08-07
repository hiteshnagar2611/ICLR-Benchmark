#!/usr/bin/env python3
"""
Test loading EVO2-40B on 5 GPUs to capture the error.
"""
import os
import time
import torch

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}", flush=True)
print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
    print(f"  GPU {i}: {props.name}, {mem_gb:.1f} GB", flush=True)

print("\nAttempting to load EVO2-40B...", flush=True)
t0 = time.time()

try:
    from evo2 import Evo2
    model = Evo2('evo2_40b')
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    # Test scoring one variant
    print("\nTest scoring 1 variant...", flush=True)
    test_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    scores = model.score_sequences([test_seq], batch_size=1, reduce_method='sum')
    print(f"Score: {scores}", flush=True)

except Exception as e:
    print(f"\n{'='*60}", flush=True)
    print(f"ERROR: {type(e).__name__}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{e}", flush=True)
    print(f"{'='*60}", flush=True)

    # Also try to get CUDA OOM details
    if "out of memory" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
        print("\nGPU memory at time of error:", flush=True)
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.1f} GB total", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
