# Admin Email — EVO2-40B GPU Request

**To:** System Admin / HPC Support
**Subject:** Request for 8x Full A100-40GB GPUs (Non-MIG) for EVO2-40B Model Evaluation

---

Dear HPC Support Team,

I am writing to request allocation of **8x full NVIDIA A100-SXM4-40GB GPUs** (non-MIG partitioned) for a time-critical benchmarking task as part of my ICLR paper research.

## What I Need

- **8x full A100-40GB GPUs** (not MIG-partitioned)
- **32 CPUs**, **256 GB RAM**
- **48-hour wall time**
- Queue: `apolloq`, project: `external`
- Host: `gnode6`

## Why Current Setup Fails

| Issue | Current | Required |
|-------|---------|----------|
| GPU memory per instance | MIG 3g.20gb = 19.5 GB | 40 GB full |
| GPU limit per user | 6 | 8 |
| Transformer Engine build | Fails (CUDA 12.4 vs cu13 mismatch) | Needs full GPU + CUDA 13 libs |

## Errors Captured (5 Job Attempts)

### Attempt 1: Job 301472 (MIG memory insufficient)
- **Error:** `RuntimeError: CUDA out of memory` — MIG 3g.20gb = 19.5GB too small for 40B model
- **PBS:** `run_evo2_40b_test.pbs`

### Attempt 2: Job 301486 (Transformer Engine build — missing nccl.h)
- **Error:** `fatal error: nccl.h: No such file or directory`
- **Fix applied:** Added NCCL include path
- **PBS:** `run_evo2_40b_te_build.pbs`

### Attempt 3: Job 301487 (Transformer Engine build — missing cudnn.h)
- **Error:** `fatal error: cudnn.h: No such file or directory`
- **Fix applied:** Added all CUDA include paths via CPATH/CFLAGS
- **PBS:** `run_evo2_40b_te_build2.pbs`

### Attempt 4: Job 301488/301490 (TE built but libcublas.so.13 not found)
- **Error:** `OSError: libcublas.so.13: cannot open shared object file: No such file or directory`
- **Root cause:** System has CUDA 12.4 (libcublas.so.12), but TE 2.17.1 + PyTorch cu128 requires CUDA 13 (libcublas.so.13)
- **PBS:** `run_evo2_40b_te_build3.pbs`, `run_evo2_40b_te_build4.pbs`

### Attempt 5: Job 301495 (Installed cu12 variant, still fails)
- **Error:** Same `libcublas.so.13` — TE torch build auto-installs `transformer_engine_cu13` because PyTorch is cu128
- **PBS:** `run_evo2_40b_final.pbs`

## Root Causes

1. **MIG Partitioning:** GPUs are partitioned as `3g.20gb` (~19.5GB each). EVO2-40B requires 8x full 40GB = 320GB total.
2. **GPU Limit:** Per-user GPU limit is 6 (`max_run_res.ngpus=[u:PBS_GENERIC=6]`). Need 8 for tensor parallelism.
3. **CUDA Version Mismatch:** System CUDA 12.4, but PyTorch 2.7.1+cu128 bundles CUDA 12.8 libraries, and Transformer Engine 2.17.1 requires CUDA 13 (`libcublas.so.13`) which doesn't exist on the system.

## Why This Matters

EVO2-40B is the largest open protein language model (40B+ parameters). Our benchmark already shows:
- EVO2-7B: ClinVar D1 AUROC = 0.836, D2 AUROC = 0.830
- Meta-Learner (ESM1b + EVO2 + physicochemical): D1 AUROC = 0.9188

Testing the 40B variant would strengthen our ICLR submission with the best available model.

## Suggested PBS Script

```bash
#!/bin/bash
#PBS -N evo2_40b_scoring
#PBS -q apolloq
#PBS -l select=1:ncpus=32:ngpus=8:mem=256gb:host=gnode6
#PBS -l walltime=48:00:00
#PBS -P external
#PBS -j oe

cd /ibdc-scratch2/home/Csir-igib001_lthukral/hitesh/bech_v4
source activate /ibdc-scratch2/home/Csir-igib001_lthukral/.conda/envs/evo2

# Set LD_LIBRARY_PATH for CUDA 13 libs (needed by Transformer Engine)
export LD_LIBRARY_PATH=/ibdc-hpc/apps1/cuda-13.0/lib64:$LD_LIBRARY_PATH

python test_evo2_40b.py
```

## Files Included

- `test_evo2_40b.py` — Test script for EVO2-40B
- `run_evo2_40b_*.pbs` — All PBS job scripts (6 files)
- `evo2_40b_*.o*` — All error log files (8 files)

Please let me know if this is feasible or if alternative arrangements are needed.

Thank you for your assistance.

Best regards,
[Your Name]
[Your Email]
[Department/Institution]
