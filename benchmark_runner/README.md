# Parallel Benchmark Runner

This toolkit allows running multiple vision-language benchmarks in parallel, providing efficient evaluation of model performance.

## Prerequisites

Before running the benchmarks, make sure you have the following dependencies installed, despite common package including `vllm` and `torch`.

```bash
pip install latex2sympy2 # MathVision.utils 
pip install Levenshtein # MathVista.calculate_score
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
The default setting for GPU allocation is one GPU for one benchmark inference. Therefore, set four gpus for running evaluation

```bash
export CKPT_PATH="/path/to/your/vprm_checkpoint"
```

2. Run all benchmarks in parallel:
```
bash run_all.sh
```
If some raw files requiring the right location are missing, it will throw errors and corresponding download path. Please follow the instructions to download them all.

### Run Individual Benchmarks
You can also run individual benchmarks separately:
```bash
# Run MathVista benchmark
bash benchmarks/mathvista.sh 9753 # mathvista requires vllm serve before running

# Run MathVision benchmark
bash benchmarks/mathvision.sh 0000 {gpu_id}

# Run MME-RealWorld-Lite benchmark
bash benchmarks/mme_realworld.sh 0000 {gpu_id}

# Run RealWorldQA benchmark
bash benchmarks/realworldqa.sh 0000 {gpu_id}

# Run ChartQA benchmark
bash benchmarks/chartqa.sh 0000 {gpu_id}
```
For the command example above, the `0000` is a placeholder indicates no need to start a vllm server before running the evaluation pipeline. For mathvista, you must start a vllm serve application before running the script.


### Reading the log files
During inference, see `logs/$BASE_DIR/logs/$INFERENCE_RUN_ID/` for the running details.

### Controlling concurrency
For mathvista, you can modify the number of `concurrency=100` to speed up inference.

## Results
After running the benchmarks, results will be available in:

- MathVista: MathVista/results/{RUN_ID}/scores_{RUN_ID}.json
- MathVision: MathVision/outputs/evaluation_results.json
- MME-RealWorld-Lite: MME-RealWorld-Lite/data/MME-RealWorld-Lite_judged_{RUN_ID}.jsonl
- RealWorldQA: realworldqa/data/RealWorldQA_judged_{RUN_ID}.jsonl
- ChartQA: chartqa/data/chartQA_${split}_judged_${INFERENCE_RUN_ID}${cot_suffix}.jsonl

Logs for each benchmark are stored in logs/{RUN_ID}/.

