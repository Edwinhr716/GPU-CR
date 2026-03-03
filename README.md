<h1 align="center">GPU-CR: GPU Checkpoint & Restore</h1>

[![cuda](https://img.shields.io/badge/CUDA-supported-brightgreen.svg?logo=nvidia)]()
[![rocm](https://img.shields.io/badge/ROCm-supported-brightgreen.svg?logo=amd)]()
[![ascend](https://img.shields.io/badge/Ascend-Developing-lightgrey.svg?logo=huawei)]()

<div align=center><img width = '150' height ='150' src ="./source/GPU-CR.png"/></div>

GPU-CR is a system designed to support efficient Checkpoint and Restore (C/R) for GPU-accelerated applications. Its key advantage is completely yielding the GPU memory of the checkpointed app (reducing VRAM usage to 0), seamlessly freeing up space for other workloads to swap in and execute.

![CLI Demonstration](./source/GPUCR_VS_CUDA.gif)
<p align="center">
  <em>A quick demonstration of executing the GPU-CR tool via the command-line interface.</em>
</p>


## I. Features

- **Cross-Vendor Support**: Experimental support for both NVIDIA and AMD GPUs.
- **Transparent C/R**: Uses `LD_PRELOAD` to inject a `vGPU` library that intercepts memory allocations and resource management.
- **Client CLI**: Simple command-line interface (`cr_client`) to trigger checkpoint and restore operations.
- **Performance Optimization**: Support for Huge Pages to accelerate memory saving.

## II. TODO

We are actively working on expanding GPU-CR's capabilities:
- 🚀 **Broader Hardware Support**: Extending compatibility to more architectures, such as Huawei Ascend.

## III. Performance Evaluation

We compare GPU-CR with existing GPU checkpoint solutions on four LLM workloads:

- Llama-8B
- Phi-4-mini-instruct
- pythia-1b
- Qwen3-1.7B

For GPU-CR, the latency is split into:

- Data — GPU data buffers
- Control — GPU control states

Total latency = Data + Control

### 1.NVIDIA (CUDA Checkpoint vs GPU-CR)
- **GPU:** NVIDIA A100-PCIE-40GB
- **Driver Version:** 580.95.05      
- **CUDA Version:** 13.0
- **vLLM Version:** 0.14.1

![Performance Comparison](./source/gpu-cr_cuda.png "NVIDIA (CUDA Checkpoint vs GPU-CR")

### 2.AMD (CRIU vs GPU-CR)
- **GPU:** AMD Instinct MI100
- **ROCm Version:** 6.4.3
- **vLLM Version:** 0.11.1-rc7
![Performance Comparison](./source/gpu-cr_criu_amd.png "AMD (CRIU vs GPU-CR)")


## IV. Prerequisites

- **Operating System**: Linux (Tested on Ubuntu 22.04).
- **Build Tools**: CMake, GCC/G++, Make.
- **Checkpoint Backend & Drivers**:
  - **NVIDIA**: 
    - Requires CUDA Toolkit 12.x or later.
    - Uses `cuda-checkpoint` (**Included in this repository**). 
    - *Note: If updates are needed, please update the parameters within the source code manually.*[[cuda-checkpoint]](https://github.com/NVIDIA/cuda-checkpoint)
  - **AMD**: 
    - Requires ROCm 6.x or later.
    - Requires a custom-built `criu` with the AMD plugin enabled. **(Manual Compilation Required)**. 
    - *Note: This custom CRIU is not included in this repository. Users must manually compile and install CRIU with AMD plugin before using GPU-CR.*[[CRIU AMDGPU Plugin Documentation]](https://github.com/checkpoint-restore/criu/blob/criu-dev/plugins/amdgpu/README.md)

## V. Building

This project utilizes CMake for building. **Please choose ONE of the following build options based on your target GPU vendor.** Do not build both simultaneously in the same environment.

### Option 1: Build for NVIDIA (CUDA)
```Bash
mkdir build && cd build
export GPU_VENDOR=NVIDIA
cmake ..
make -j$(nproc)
```
This generates `vGPU-NVIDIA.so` and `cr_client`.

### Option 2: Build for AMD (ROCm)

```bash
mkdir build && cd build
export GPU_VENDOR=AMD
cmake ..
make -j$(nproc)
```
This generates `vGPU-AMD.so` and `cr_client`.

## VI. Usage

### 1. Environment Configuration
Before running, configure the necessary environment variables.

#### (1) General Configuration (Both NVIDIA & AMD)
- VRAM Storage Strategy
By default, GPU memory is saved to Huge Pages. You can optionally save it to a file system path using EXPORT_FILE_PATH.

```Bash
# Optional: Path to save video memory content as a file.
# If NOT set, the system defaults to saving VRAM to Huge Pages.
export EXPORT_FILE_PATH=/path/to/save/vram_dump_path
```

- Huge Pages (Recommended for Acceleration)
Huge pages can significantly accelerate the save process for both vendors.

```Bash
# Example: reserve 80GB huge pages
sudo bash -c "echo 40960 > /proc/sys/vm/nr_hugepages"

sudo mkdir /mnt/huge-ckpt
sudo mount -t hugetlbfs nodev /mnt/huge-ckpt
sudo chmod 777 -R /mnt/huge-ckpt
```

#### (2) AMD-Specific Configuration
If you are using AMD GPUs, you must specify the directory where CRIU will store its checkpoint files.
```Bash
export AMD_CKPT_DIR=/path/to/save/criu_files
```

### 2. Running an Application

Launch the target application (e.g., a Python script using PyTorch/vLLM or a C++ binary) using `LD_PRELOAD`.

**(1) Example (NVIDIA):**
```bash
LD_PRELOAD=/path/to/build/vGPU-NVIDIA.so python3 ./apps/vllm/serving_vllm_nvidia.py
```

**(2) Example (AMD):**
```bash
LD_PRELOAD=/path/to/build/vGPU-AMD.so ./apps/vllm/serving_vllm_amd.sh
```

### 3. Checkpointing

Use the `cr_client` tool to trigger a checkpoint.

```bash
# -i: initialization mode
# -c: Checkpoint mode
# -p: Target PID
# -m: (Optional) The PID of the original parent process (Master) that CRIU needs to control.(for CRIU in AMD mode)
./cr_client -c -p <TARGET_PID>
# or
./cr_client -c -p <GPU_CHILD_PID> -m <PARENT_PID>
```

### 4. Restoring

Restore the process from the checkpoints.

```bash
# -r: Restore mode
# -p: Target PID (the original PID)
./cr_client -r -p <TARGET_PID>
```

## VII. Directory Structure

- `src/`: Source code for the vGPU library and cr_client.
  - `GPUs/NVIDIA/`: NVIDIA-specific implementation (CUDA hooks).
  - `GPUs/AMD/`: AMD-specific implementation (HIP hooks).
  - `cr_client.cpp`: Control client implementation.
- `apps/`: Example scripts and applications (e.g., vLLM examples).

## VIII. Citation

This project is based on our paper:

```bibtex
@inproceedings{GCR,
  author    = {Shaoxun Zeng and Tingxu Ren and Jiwu Shu and Youyou Lu},
  title     = {GPU Checkpoint/Restore Made Fast and Lightweight},
  booktitle = {24rd USENIX Conference on File and Storage Technologies (FAST'26)},
  year      = {2026},
  address   = {Santa Clara, CA},
  month     = feb,
  publisher = {USENIX Association},
  url       = {https://www.usenix.org/conference/fast26/presentation/zeng}
}
```

And the implementation of the paper is in: https://github.com/thustorage/GCR
