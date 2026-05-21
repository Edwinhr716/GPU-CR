export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=0
export GPU_VENDOR=AMD
export VLLM_Model="${VLLM_MODEL:-/path/to/your/model}"

echo "Starting vLLM API Server..."
echo "Current Shell PID: $$ (This will become the Python PID)"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model $VLLM_Model \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 2048