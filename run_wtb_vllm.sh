#!/bin/bash
#SBATCH -p solgpu                 # Partition (job queue)
#SBATCH -G 1                      # Number of GPUs
#SBATCH -c 8                      # Number of CPU cores
#SBATCH --mem=40G                 # Job memory request
#SBATCH -t 4:00:00                # Time limit (hrs:min:sec)
#SBATCH -o wtb_vllm_%j.log        # Standard output and error log

# 1. Load Anaconda and activate the environment
module load mamba/conda
conda activate wtb_vllm

# 2. Choose model (defaults to Qwen/Qwen2.5-7B-Instruct if no argument is passed)
MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
echo "==== Starting vLLM Server for model: $MODEL ===="

# 3. Start vLLM OpenAI-Compatible API Server in the background
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port 8000 \
  --dtype auto &
VLLM_PID=$!

# 4. Wait for the vLLM server to start up (poll the model check endpoint)
echo "Waiting for vLLM server to boot up on port 8000..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
  sleep 5
done
echo "vLLM server is online!"

# 5. Navigate to the benchmark script directory
cd WildToolBench/wild-tool-bench

# 6. Run the WildToolBench evaluation
echo "==== Running WildToolBench Inference ===="
python3 -u -m wtb.openfunctions_evaluation --model "$MODEL" --num-threads 4

echo "==== Running WildToolBench Evaluation / Scorer ===="
python3 -u -m wtb.eval_runner --model "$MODEL"

# 7. Clean up and shut down the vLLM server background process
kill $VLLM_PID
echo "==== Job Completed ===="
