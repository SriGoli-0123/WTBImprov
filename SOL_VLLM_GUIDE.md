# Running WildToolBench with vLLM on ASU's Research Computing Platform (SOL)

This guide walks you through setting up a Conda environment, launching a vLLM server on a GPU node, and executing the WildToolBench evaluation benchmark.

---

## 📂 What this Workspace Contains
- **Auto-Fallbacks for vLLM:** We modified `wtb/model_handler/handler_map.py` to use a `defaultdict`. Any Hugging Face model loaded via vLLM (e.g. `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`) will automatically fallback to the `OpenAIHandler` without needing manual code mapping edits.
- **Preconfigured Endpoints:** The `.env` file points to `http://localhost:8000/v1` (the default port for vLLM's OpenAI API server).
- **Automation Script:** [run_wtb_vllm.sh](file:///Users/sriharshithgoli/Desktop/wildtoolbench_workspace_vllm/run_wtb_vllm.sh) is a Slurm batch file that requests a GPU node, starts the vLLM server, polls until it is ready, runs the evaluations, and cleans up.

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

## 🚀 Execution Methods on SOL

### Method A: Automated Slurm Batch Run (Recommended)
This uses the Slurm workload manager to queue your job, request a GPU node, and execute the benchmark fully in the background.

Submit the script with:
```bash
sbatch run_wtb_vllm.sh <huggingface_model_id>
```
*Example (default):*
```bash
sbatch run_wtb_vllm.sh Qwen/Qwen2.5-7B-Instruct
```
*Example (70B model - may require more GPU memory, e.g. a larger partition/GRES):*
```bash
sbatch run_wtb_vllm.sh meta-llama/Llama-3.1-70B-Instruct
```
Monitor the outputs in real-time by viewing the generated log:
```bash
tail -f wtb_vllm_*.log
```

---

### Method B: Interactive GPU Session Run
If you want to run steps manually, debug, or interact with the server, request an interactive GPU allocation:

1. **Request a GPU Node:**
   ```bash
   srun -p solgpu -G 1 -c 8 --mem=40G -t 4:00:00 --pty bash
   ```
2. **Activate Environment & Start vLLM:**
   Once logged into the GPU compute node, start the server in the background:
   ```bash
   module load mamba/conda
   conda activate wtb_vllm
   
   python3 -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-7B-Instruct \
     --port 8000 \
     --dtype auto &
   ```
   *Wait for the log to output: `Uvicorn running on http://127.0.0.1:8000`*
3. **Execute Benchmark Scripts:**
   ```bash
   cd WildToolBench/wild-tool-bench
   
   # 1. Run inference (makes API calls to vLLM)
   python3 -u -m wtb.openfunctions_evaluation --model Qwen/Qwen2.5-7B-Instruct --num-threads 4
   
   # 2. Score output and save metrics
   python3 -u -m wtb.eval_runner --model Qwen/Qwen2.5-7B-Instruct
   ```
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
