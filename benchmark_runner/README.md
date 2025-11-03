# Parallel Benchmark Runner

This toolkit allows running multiple vision-language benchmarks in parallel, providing efficient evaluation of model performance. All benchmark scripts support command-line arguments to override configuration settings, making it easy to test different models and settings without editing configuration files.

## Prerequisites

Before running the benchmarks, make sure you have the following dependencies installed:

```bash
pip install latex2sympy2  # MathVision.utils 
pip install Levenshtein    # MathVista.calculate_score
```

## File Structure

- `config.sh`: Contains common configuration variables (checkpoint path, GPU list, API keys, etc.)
- `check_raw_files.sh`: Checks and downloads required raw data files
- `run_all.sh`: Main script to run all benchmarks in parallel with one command
- `benchmarks/`: Directory containing individual benchmark scripts:
  - `mathvista.sh`: MathVista benchmark runner
  - `mathvision.sh`: MathVision benchmark runner
  - `mme_realworld.sh`: MME-RealWorld-Lite benchmark runner
  - `realworldqa.sh`: RealWorldQA benchmark runner
  - `blink.sh`: BLINK benchmark runner
  - `chartqa.sh`: ChartQA benchmark runner
  - `mmstar.sh`: MMStar benchmark runner

---

## Configuration

### Basic Configuration (`config.sh`)

Edit `config.sh` to set up your environment:

```bash
# Set your model checkpoint path
export CKPT_PATH="/path/to/your/model/checkpoint"

# Set available GPUs (comma-separated list)
# IMPORTANT: You need at least as many GPUs as benchmarks you want to run in parallel
# Each benchmark requires one dedicated GPU
export CUDA_DEVICES_LIST="4,5,6,7,8,9,10"  # Example for 7 GPUs

# API endpoints for evaluation (judge models)
export CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT="https://your-api-endpoint/v1"
export CUSTOMIZED_REMOTE_OPENAI_API_KEY="your-api-key"
```

**⚠️ GPU Requirement**: The `run_all.sh` script automatically assigns one GPU per benchmark. If you want to run all 7 benchmarks in parallel, you need to configure at least 7 GPUs in `CUDA_DEVICES_LIST`. The script will automatically check and warn you if there are not enough GPUs.

---

## Usage

### Option 1: One-Click Run All Benchmarks

The easiest way to run all benchmarks in parallel:

```bash
# Run all benchmarks with default settings from config.sh
bash run_all.sh

# Run with custom checkpoint path
bash run_all.sh --ckpt-path /path/to/different/model

# Run only selected benchmarks (useful for testing or not enough GPUs)
bash run_all.sh --benchmarks mathvista,blink,mathvision

# Combine options
bash run_all.sh --ckpt-path /path/to/model --benchmarks mathvista,blink --run-id my_custom_run_id
```

#### `run_all.sh` Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `-k, --ckpt-path <path>` | Override checkpoint path from config.sh | `--ckpt-path /path/to/model` |
| `-i, --run-id <id>` | Manually specify a run ID (default: auto-generated) | `--run-id my_run_20241103` |
| `-b, --benchmarks <list>` | Comma-separated list of benchmarks to run | `--benchmarks mathvista,blink` |
| `-h, --help` | Show help message | `--help` |

**Available benchmarks**: `mathvista`, `mathvision`, `mme_realworld`, `realworldqa`, `blink`, `chartqa`, `mmstar`

#### How It Works

- **Automatic GPU Assignment**: Each benchmark is assigned one dedicated GPU automatically
- **Automatic Port Assignment**: Benchmarks that require VLLM serve get unique ports (9751, 9752, etc.)
- **Parallel Execution**: All benchmarks run in parallel as background processes
- **Smart Resource Management**: 
  - Benchmarks requiring VLLM serve: `mathvista`, `blink`
  - Benchmarks using direct inference: `mathvision`, `mme_realworld`, `realworldqa`, `chartqa`, `mmstar`

---

### Option 2: Run Individual Benchmarks

You can also run each benchmark script separately with full command-line argument support:

#### Common Arguments (All Benchmarks)

All benchmark scripts support these common arguments:

| Option | Description | Default |
|--------|-------------|---------|
| `-k, --ckpt-path <path>` | Model checkpoint path | From `config.sh` |
| `-g, --gpu-id <id>` | GPU ID to use | `0` |
| `-i, --run-id <id>` | Run ID | Auto-generated |
| `-c, --use-cot <0\|1>` | Use Chain of Thought (1=yes, 0=no) | `1` |
| `-p, --port <port>` | VLLM inference port (if needed) | From `config.sh` |
| `-h, --help` | Show help message | - |

#### Benchmark-Specific Examples

**MathVista** (requires VLLM serve):
```bash
# Use default settings
bash benchmarks/mathvista.sh

# Override settings
bash benchmarks/mathvista.sh \
  --ckpt-path /path/to/model \
  --gpu-id 0 \
  --port 9753 \
  --use-cot 1 \
  --num-threads 100

# MathVista also supports:
#   -n, --num-threads <num>    Number of concurrent threads (default: 100)
```

**MathVision** (direct inference, no serve needed):
```bash
bash benchmarks/mathvision.sh \
  --ckpt-path /path/to/model \
  --gpu-id 1 \
  --use-cot 0 \
  --subset testmini

# MathVision also supports:
#   -s, --subset <subset>      'testmini' or 'test' (default: testmini)
```

**BLINK** (requires VLLM serve):
```bash
bash benchmarks/blink.sh \
  --ckpt-path /path/to/model \
  --gpu-id 2 \
  --port 9751 \
  --use-cot 1 \
  --task-name all

# BLINK also supports:
#   -t, --task-name <name>     Task name (default: all)
```

**ChartQA** (direct inference):
```bash
bash benchmarks/chartqa.sh \
  --ckpt-path /path/to/model \
  --gpu-id 3 \
  --use-cot 1 \
  --split test

# ChartQA also supports:
#   -s, --split <split>        'test' or 'val' (default: test)
```

**MME-RealWorld-Lite** (direct inference):
```bash
bash benchmarks/mme_realworld.sh \
  --ckpt-path /path/to/model \
  --gpu-id 4 \
  --use-cot 1 \
  --image-base-dir /path/to/images

# MME-RealWorld-Lite also supports:
#   -b, --image-base-dir <dir> Base directory for images
```

**RealWorldQA** (direct inference):
```bash
bash benchmarks/realworldqa.sh \
  --ckpt-path /path/to/model \
  --gpu-id 5 \
  --use-cot 1
```

**MMStar** (direct inference):
```bash
bash benchmarks/mmstar.sh \
  --ckpt-path /path/to/model \
  --gpu-id 6 \
  --use-cot 1
```

#### Viewing Help for Any Script

Each script has built-in help:

```bash
bash benchmarks/mathvista.sh --help
bash benchmarks/chartqa.sh --help
# etc.
```

---

## Benchmark Dependencies

### Benchmarks Requiring VLLM Serve

These benchmarks need a VLLM inference server running:

- **MathVista**: Uses `--inference_api` parameter
- **BLINK**: Uses `INFERENCE_ENDPOINT` environment variable

When running these via `run_all.sh`, the VLLM server is automatically started and managed. When running individually, you must either:
1. Start the VLLM server manually before running, OR
2. The script will attempt to check if the server is healthy (BLINK only)

### Benchmarks Using Direct Inference

These benchmarks run inference directly (no serve needed):

- **MathVision**: Direct model inference
- **MME-RealWorld-Lite**: Direct model inference
- **RealWorldQA**: Direct model inference
- **ChartQA**: Direct model inference
- **MMStar**: Direct model inference

---

## Monitoring Progress

### Log Files

During execution, logs are stored in:
```
logs/{RUN_ID}/
```

Each benchmark has its own log file:
- `{benchmark}_run.log`: Main execution log
- `{benchmark}_inference.log`: Inference process log
- `vllm_server_{port}.log`: VLLM server log (if applicable)

### Real-Time Monitoring

You can monitor progress in real-time:

```bash
# Watch all log files
tail -f logs/{RUN_ID}/*.log

# Watch a specific benchmark
tail -f logs/{RUN_ID}/mathvista_run.log
```

---

## Results

After running the benchmarks, results will be available in:

- **MathVista**: `MathVista/results/{RUN_ID}/scores_{RUN_ID}.json`
- **MathVision**: `MathVision/data/MathVision-{subset}_judged_{RUN_ID}.jsonl`
- **MME-RealWorld-Lite**: `MME-RealWorld-Lite/data/MME-RealWorld-Lite_judged_{RUN_ID}.jsonl`
- **RealWorldQA**: `realworldqa/data/RealWorldQA_judged_{RUN_ID}.jsonl`
- **BLINK**: `BLINK_Benchmark/eval/results/`
- **ChartQA**: `ChartQA/data/chartQA_{split}_judged_{RUN_ID}.jsonl`
- **MMStar**: `MMStar/data/mmstar_judged_{RUN_ID}.jsonl`

All logs are stored in: `logs/{RUN_ID}/`

---

## Troubleshooting

### "Not enough GPUs" Error

If you see this error when running `run_all.sh`:
```
Error: Not enough GPUs configured!
  Required: 7 GPUs (one per benchmark)
  Configured: 5 GPUs in CUDA_DEVICES_LIST
```

**Solution**: Update `config.sh` to include enough GPUs:
```bash
export CUDA_DEVICES_LIST="4,5,6,7,8,9,10"  # Add more GPUs
```

Or run fewer benchmarks:
```bash
bash run_all.sh --benchmarks mathvista,blink,mathvision  # Only 3 benchmarks
```

### VLLM Server Not Starting

If VLLM servers fail to start:
1. Check the server log: `logs/{RUN_ID}/vllm_server_{port}.log`
2. Ensure the GPU is not already in use
3. Check if the port is already in use: `netstat -tuln | grep {port}`

### Raw Data Files Missing

If you see errors about missing data files, the script will automatically show download instructions. Follow the instructions to download the required files.

---

## Tips

1. **Test Individual Scripts First**: Before running all benchmarks, test individual scripts to ensure your setup is correct:
   ```bash
   bash benchmarks/mathvista.sh --help
   bash benchmarks/mathvista.sh --gpu-id 0 --use-cot 0
   ```

2. **Use Run ID for Organization**: Use `--run-id` to organize your runs:
   ```bash
   bash run_all.sh --run-id experiment_001 --benchmarks mathvista,blink
   ```

3. **Monitor Resource Usage**: When running in parallel, monitor GPU and memory usage:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Incremental Testing**: Start with a few benchmarks to test your setup:
   ```bash
   bash run_all.sh --benchmarks mathvista,mathvision
   ```

---

## Advanced Usage

### Custom Configuration Override

You can override any configuration via command-line without editing `config.sh`:

```bash
# Run with different checkpoint
bash run_all.sh --ckpt-path /path/to/model_v2

# Run without CoT
bash benchmarks/mathvista.sh --use-cot 0

# Run with custom concurrency
bash benchmarks/mathvista.sh --num-threads 200
```

### Running Different Splits/Subsets

Some benchmarks support different data splits:

```bash
# ChartQA: test or val
bash benchmarks/chartqa.sh --split val

# MathVision: testmini or test
bash benchmarks/mathvision.sh --subset test
```

---

## Summary

- ✅ **Command-line arguments** override configuration for all scripts
- ✅ **One-click execution** with `run_all.sh`
- ✅ **Automatic resource allocation** (GPU and port assignment)
- ✅ **Parallel execution** of all benchmarks
- ✅ **Flexible selection** of which benchmarks to run
- ✅ **Easy debugging** with comprehensive logging

For help on any script, use `--help` flag:
```bash
bash benchmarks/{benchmark_name}.sh --help
bash run_all.sh --help
```
