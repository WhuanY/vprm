# Parallel Benchmark Runner

This toolkit allows running multiple vision-language benchmarks in parallel, providing efficient evaluation of model performance.

## Prerequisites

Before running the benchmarks, make sure you have the following dependencies installed, despite common package including `vllm` and `torch`

```bash
pip install latex2sympy2 # MathVision.utils 
pip install Levenshtein # MathVista.calculate_score
# Other dependencies as required by individual benchmarks
```

## File Structure

- `config.sh`: Contains common configuration variables  
- `check_raw_files.sh`: Checks and downloads required raw data files  
- `run_all.sh`: Main script to run all benchmarks in parallel  
- `benchmarks/`: Directory containing individual benchmark scripts:  
  - `mathvista.sh`: MathVista benchmark runner  
  - `mathvision.sh`: MathVision benchmark runner  
  - `mme_realworld.sh`: MME-RealWorld-Lite benchmark runner  
  - `realworldqa.sh`: RealWorldQA benchmark runner  

---

## Usage

### One-Click Run All Benchmarks

1. Edit `config.sh` to set your model checkpoint path:
The default setting for GPU allocation is one GPU for one benchmark inference . Therefore, set four gpus for running evaluation

```bash
export CKPT_PATH="/path/to/your/vprm_checkpoint"
```

2. Run all benchmarks in parallel:
```
bash run_all.sh
```
If some raw files requiring the right location are missing, it will throw errors and corresponding installation path. Please follow the instructions to download them all.

### Run Individual Benchmarks
You can also run individual benchmarks separately:
```bash
# Run MathVista benchmark
bash benchmarks/mathvista.sh

# Run MathVision benchmark
bash benchmarks/mathvision.sh

# Run MME-RealWorld-Lite benchmark
bash benchmarks/mme_realworld.sh

# Run RealWorldQA benchmark
bash benchmarks/realworldqa.sh
```

### Reading the log files
During inference, see `logs/$BASE_DIR/logs/$INFERENCE_RUN_ID/` for the running details.

## Results
After running the benchmarks, results will be available in:

- MathVista: MathVista/results/{RUN_ID}/scores_{RUN_ID}.json
- MathVision: MathVision/outputs/evaluation_results.json
- MME-RealWorld-Lite: MME-RealWorld-Lite/data/MME-RealWorld-Lite_judged_{RUN_ID}.jsonl
- RealWorldQA: realworldqa/data/RealWorldQA_judged_{RUN_ID}.jsonl
Logs for each benchmark are stored in logs/{RUN_ID}/.

## Resource Management
The parallel execution uses separate VLLM server instances for each benchmark, each running on a different port. This approach maximizes resource utilization while preventing conflicts between benchmarks.

Each VLLM server is automatically started before running a benchmark and shut down after completion.
