# SPDX-License-Identifier: Apache-2.0
"""
serving_vllm_multi_gpu.py — Multi-GPU vLLM inference server with CR support

This script starts a vLLM inference server using tensor parallelism across
multiple GPUs.  It is designed to work with the multi-GPU CR system:

  1. Launch with launch_multi_gpu.sh (sets LD_PRELOAD + NCCL_CUMEM_ENABLE=1)
  2. vGPU.so automatically hooks cudaMalloc + ncclCommInitRank in ALL workers
  3. Use multi_cr_client to checkpoint/restore ALL worker processes

Usage:
  # 4-GPU tensor parallel (Llama 70B needs >=4 GPUs)
  ./launch_multi_gpu.sh python serving_vllm_multi_gpu.py --tp 4

  # 2-GPU tensor parallel
  ./launch_multi_gpu.sh python serving_vllm_multi_gpu.py --tp 2

  # Then checkpoint:
  PIDS=$(pgrep -f "serving_vllm_multi_gpu" | tr '\\n' ',')
  ./build/multi_cr_client -c -p ${PIDS%,}

  # And restore:
  ./build/multi_cr_client -r -p ${PIDS%,}

Note:
  - enforce_eager=True is REQUIRED to disable CUDA Graphs (incompatible with
    cuda-checkpoint).
  - NCCL_CUMEM_ENABLE=1 must be set BEFORE vLLM starts (launch_multi_gpu.sh
    does this automatically).
"""

import argparse
import os
import socket
import time

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU vLLM CR Server")
    parser.add_argument("--model", type=str,
                        default=os.getenv("VLLM_MODEL", "/path/to/your/model"),
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--pp", type=int, default=2,
                    help="Pipeline parallel size (number of GPUs)")
    parser.add_argument("--port", type=int, default=10000,
                        help="UDP port to listen on")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum model context length")
    parser.add_argument("--gpu-util", type=float,
                        default=float(os.getenv("VLLM_GPU_UTIL", "0.5")),
                        help="gpu_memory_utilization (0.0-1.0)")
    args = parser.parse_args()

    print(f"[Multi-GPU vLLM] Starting with TP={args.tp}, model={args.model}")
    print(f"[Multi-GPU vLLM] PID={os.getpid()}")
    print(f"[Multi-GPU vLLM] NCCL_CUMEM_ENABLE={os.environ.get('NCCL_CUMEM_ENABLE', 'NOT SET')}")
    print(f"[Multi-GPU vLLM] LD_PRELOAD={os.environ.get('LD_PRELOAD', 'NOT SET')}")

    # Create sampling params
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

    # Create LLM with tensor parallelism
    # enforce_eager=True is CRITICAL: disables CUDA Graphs which are
    # incompatible with cuda-checkpoint state freezing
    # disable_custom_all_reduce=True: our cudaMalloc hook uses VMM
    # (cuMemCreate/cuMemMap) which is incompatible with cudaIpcGetMemHandle
    # that custom allreduce relies on. Disabling falls back to NCCL allreduce.
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=args.gpu_util,
    )
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    print(f"[Multi-GPU vLLM] Model loaded successfully with TP={args.tp}")
    print(f"[Multi-GPU vLLM] To checkpoint, run:")
    print(f"  PIDS=$(pgrep -f 'serving_vllm_multi_gpu' | tr '\\n' ',')")
    print(f"  ./build/multi_cr_client -i -p ${{PIDS%,}}")
    print(f"  ./build/multi_cr_client -c -p ${{PIDS%,}}")

    # UDP server for inference requests
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('0.0.0.0', args.port))
    print(f"[Multi-GPU vLLM] UDP listening on port {args.port}...")

    while True:
        data, addr = udp_socket.recvfrom(4096)
        prompt = data.decode()
        print(f"\n[Request] From {addr}: {prompt[:100]}...")

        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = (time.time() - start) * 1000

        generated_text = outputs[0].outputs[0].text
        print(f"[Response] {elapsed:.1f}ms, {len(generated_text)} chars")
        print(f"  Output: {generated_text[:200]}...")

        udp_socket.sendto(generated_text.encode()[:4096], addr)


if __name__ == "__main__":
    main()
