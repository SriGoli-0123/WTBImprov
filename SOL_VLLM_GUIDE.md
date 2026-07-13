# Running WildToolBench with vLLM on ASU's Research Computing Platform (SOL)

This guide walks you through setting up a Conda environment, launching a vLLM server on a GPU node, and executing the WildToolBench evaluation benchmark.

---

## 📂 What this Workspace Contains
- **Auto-Fallbacks for vLLM:** We modified `wtb/model_handler/handler_map.py` to use a `defaultdict`. Any Hugging Face model loaded via vLLM (e.g. `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`) will automatically fallback to the `OpenAIHandler` without needing manual code mapping edits.
- **Preconfigured Endpoints:** The `.env` file points to `http://localhost:8000/v1` (the default port for vLLM's OpenAI API server).
- **Consensus decoding + triage prompt + Session Ledger:** The improvement method (see `IMPROVEMENT_METHOD.md`). Knobs: `WTB_SC_N` (candidates/step, default 5; `1` disables voting), `WTB_SC_TEMP` (diversity temperature, default 0.8), `WTB_SYS_MODE` (`triage` default / `minimal` = original bare date prompt), `WTB_LEDGER` (`1` default = terse fact ledger injected next to the current turn / `0` off).
- **`patch_vllm.py`:** Patches vLLM's `hermes_tool_parser` so it can extract multiple/parallel tool calls from one response. Run it once after installing vLLM (and after any vLLM reinstall). The handler also recovers/abstains client-side, so occasional `[vLLM Patch Warning] Failed to parse tool call...` lines are harmless (a truncated generation that would fail anyway).

---

## 🛠️ Step-by-Step Setup on SOL

### Step 1: Get the Workspace onto SOL
Either upload a compressed copy, or clone/pull the git repo directly - pick one:

**Option A - tar/scp (copies every file, including `.env`):**
```bash
# On your local terminal (compress)
tar -czvf wildtoolbench_workspace_vllm.tar.gz wildtoolbench_workspace_vllm/

# Upload to SOL (replace with your ASU ASURITE username)
scp wildtoolbench_workspace_vllm.tar.gz username@sol.asu.edu:~/

# On your SOL terminal (decompress)
tar -xzvf wildtoolbench_workspace_vllm.tar.gz
cd wildtoolbench_workspace_vllm/
```

**Option B - `git clone` / `git pull` (recommended for iterating on code):**
```bash
git clone https://github.com/SriGoli-0123/WTBImprov.git wildtoolbench_workspace_vllm
cd wildtoolbench_workspace_vllm/
git checkout demo   # or: git pull origin demo, if already cloned
```
⚠️ `.env` is intentionally excluded from git (`.gitignore`) so real API keys never
get committed - `git pull` will never create it for you. One-time per machine:
```bash
cd WildToolBench/wild-tool-bench
cp .env.example .env
```
(The default values already point at the local vLLM server this guide starts, so
no editing is needed unless you're using a different host/port or a real API key.)

### Step 2: Create and Configure the Conda Environment
Load ASU SOL's Anaconda module and create a dedicated environment:
```bash
# Load conda
module load mamba/conda

# Create python 3.10 env
conda create -n wtb_vllm python=3.10 -y
conda activate wtb_vllm

# Install vLLM (GPU-accelerated LLM serving framework)
pip install vllm

# Install benchmark requirements
cd WildToolBench
pip install -r requirements.txt
cd ..
```

---

## 🚀 Running the Benchmark (Interactive GPU Session)

1. **Request a GPU Node:**
   ```bash
   srun -p solgpu -G 1 -c 8 --mem=40G -t 6:00:00 --pty bash
   ```
   (Consensus decoding samples 5 candidates/step, so allow ~2x the baseline wall time.)

2. **Activate Environment & Start vLLM:**
   Once logged into the GPU compute node:
   ```bash
   module load mamba/conda
   conda activate wtb_vllm

   # One-time (and after any vLLM reinstall): patch the tool-call parser
   python3 patch_vllm.py

   python3 -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-7B-Instruct \
     --port 8000 \
     --dtype auto \
     --enable-auto-tool-choice \
     --tool-call-parser hermes &
   ```
   *Wait for the log to output: `Uvicorn running on http://127.0.0.1:8000`.*
   (For Llama models use `--tool-call-parser llama3_json` instead of `hermes`.)

3. **Execute Benchmark Scripts:**
   ```bash
   cd WildToolBench/wild-tool-bench
   cp .env.example .env          # one-time: git does not carry the gitignored .env

   # Consensus decoding is ON by default (WTB_SC_N=5). Results go to a separate
   # folder so the committed baseline in result/ + score/ is kept for comparison.
   # 1. Run inference (makes API calls to vLLM)
   python3 -u -m wtb.openfunctions_evaluation \
     --model Qwen/Qwen2.5-7B-Instruct --num-threads 4 --result-dir result_consensus

   # 2. Score output and save metrics (note: underscore form of the model name)
   python3 -u -m wtb.eval_runner \
     --model Qwen_Qwen2.5-7B-Instruct --result-dir result_consensus --score-dir score_consensus
   ```
   *Ablation — prompt only, voting off:* prefix inference with `WTB_SC_N=1` and use
   fresh dirs (e.g. `--result-dir result_prompt_only` / `--score-dir score_prompt_only`),
   because generation resumes from and skips any ids already present in a result dir.

4. **Shutdown Server:**
   ```bash
   kill %1
   ```

---

## 📊 Result Extraction on SOL
You can run the interactive search tool directly on SOL to extract scenario data or debug failures:
```bash
python3 scratch/extract_scenario.py --model Qwen/Qwen2.5-7B-Instruct --error-only
```
