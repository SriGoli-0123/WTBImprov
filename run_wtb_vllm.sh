#!/bin/bash
#SBATCH -p solgpu                 # Partition (job queue)
#SBATCH -G 1                      # Number of GPUs
#SBATCH -c 8                      # Number of CPU cores
#SBATCH --mem=40G                 # Job memory request
#SBATCH -t 6:00:00                # Time limit (consensus decoding samples 5 candidates/step, ~2x baseline wall time)
#SBATCH -o wtb_vllm_%j.log        # Standard output and error log

# 1. Load Anaconda and activate the environment
module load mamba/conda
conda activate wtb_vllm

# Apply JSON safety patch to the vLLM server's hermes_tool_parser to prevent server-side crashes on malformed candidate outputs
if [ -f "patch_vllm.py" ]; then
  python3 patch_vllm.py
fi

# 2. Choose model (defaults to Qwen/Qwen2.5-7B-Instruct if no argument is passed)
MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
# Tool-call parser for vLLM: hermes works for Qwen2.5/Qwen3;
# pass llama3_json as the 2nd arg for Llama-3.x models.
TOOL_PARSER=${2:-"hermes"}
echo "==== Starting vLLM Server for model: $MODEL (tool parser: $TOOL_PARSER) ===="

# Consensus decoding (self-consistency) config, read by wtb/model_handler/base_handler.py:
#   WTB_SC_N=1 disables voting (reproduces single-sample behavior)
export WTB_SC_N=${WTB_SC_N:-5}      # candidates sampled per step (1 anchor + N-1 diverse)
export WTB_SC_TEMP=${WTB_SC_TEMP:-0.8}  # temperature of the diversity samples

# 3. Start vLLM OpenAI-Compatible API Server in the background
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port 8000 \
  --dtype auto \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER" &
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
# NOTE: results go to result_consensus/ + score_consensus/ so the committed
# baseline in result/ + score/ is kept intact for comparison. Generation
# resumes from existing entries, so a timed-out job can simply be resubmitted.
echo "==== Running WildToolBench Inference (consensus decoding, N=$WTB_SC_N) ===="
python3 -u -m wtb.openfunctions_evaluation --model "$MODEL" --num-threads 4 --result-dir result_consensus

echo "==== Running WildToolBench Evaluation / Scorer ===="
# The result folder replaces "/" with "_" in the model name; eval_runner matches
# on the folder name, so pass the underscore form.
MODEL_DIR=${MODEL//\//_}
python3 -u -m wtb.eval_runner --model "$MODEL_DIR" --result-dir result_consensus --score-dir score_consensus

# 7. Clean up and shut down the vLLM server background process
kill $VLLM_PID
echo "==== Job Completed ===="
