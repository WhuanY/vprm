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
- **Parallel Execution**: All benchmarks run in parallel as background processes
- **Batch Processing**: All benchmarks use batch processing for efficient inference

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
| `-b, --bz <size>` | Batch size for inference | Benchmark-specific (see below) |
| `-h, --help` | Show help message | - |

#### Benchmark-Specific Examples

**MathVista**:
```bash
# Use default settings
bash benchmarks/mathvista.sh

# Override settings
bash benchmarks/mathvista.sh \
  --ckpt-path /path/to/model \
  --gpu-id 0 \
  --use-cot 1 \
  --bz 20

# MathVista also supports:
#   -b, --bz <size>            Batch size for inference (default: 100)
```

**MathVision**:
```bash
bash benchmarks/mathvision.sh \
  --ckpt-path /path/to/model \
  --gpu-id 1 \
  --use-cot 0 \
  --subset testmini \
  --bz 50

# MathVision also supports:
#   -s, --subset <subset>      'testmini' or 'test' (default: testmini)
#   -b, --bz <size>            Batch size for inference (default: 100)
```

**BLINK**:
```bash
bash benchmarks/blink.sh \
  --ckpt-path /path/to/model \
  --gpu-id 2 \
  --use-cot 1 \
  --task-name all \
  --bz 50

# BLINK also supports:
#   -t, --task-name <name>     Task name (default: all)
#   -b, --bz <size>            Batch size for inference (default: 50)
```

**ChartQA**:
```bash
bash benchmarks/chartqa.sh \
  --ckpt-path /path/to/model \
  --gpu-id 3 \
  --use-cot 1 \
  --split test \
  --bz 50

# ChartQA also supports:
#   -s, --split <split>        'test' or 'val' (default: test)
#   -b, --bz <size>            Batch size for inference (default: 50)
```

**MME-RealWorld-Lite**:
```bash
bash benchmarks/mme_realworld.sh \
  --ckpt-path /path/to/model \
  --gpu-id 4 \
  --use-cot 1 \
  --bz 50

# MME-RealWorld-Lite also supports:
#   -b, --bz <size>            Batch size for inference (default: 50)
# Note: Image base directory is hardcoded in the script
```

**RealWorldQA**:
```bash
bash benchmarks/realworldqa.sh \
  --ckpt-path /path/to/model \
  --gpu-id 5 \
  --use-cot 1 \
  --bz 50

# RealWorldQA also supports:
#   -b, --bz <size>            Batch size for inference (default: 50)
```

**MMStar**:
```bash
bash benchmarks/mmstar.sh \
  --ckpt-path /path/to/model \
  --gpu-id 6 \
  --use-cot 1 \
  --bz 50

# MMStar also supports:
#   -b, --bz <size>            Batch size for inference (default: 50)
```

#### Viewing Help for Any Script

Each script has built-in help:

```bash
bash benchmarks/mathvista.sh --help
bash benchmarks/chartqa.sh --help
# etc.
```

---

## Benchmark Architecture

### Direct VLLM Model Loading

All benchmarks now use **direct VLLM model loading** with batch processing:

- **No External Server Required**: All benchmarks load the model directly using `vllm.LLM`
- **Batch Processing**: Efficient batch inference for better GPU utilization
- **Automatic Resource Management**: Each benchmark manages its own VLLM instance
- **Consistent Interface**: All benchmarks use the same batch processing approach

### Batch Size Configuration

Each benchmark has a default batch size that can be customized:

- **MathVista**: Default batch size 100
- **MathVision**: Default batch size 100
- **BLINK**: Default batch size 50
- **ChartQA**: Default batch size 50
- **MME-RealWorld-Lite**: Default batch size 50
- **RealWorldQA**: Default batch size 50
- **MMStar**: Default batch size 50

Adjust batch size based on your GPU memory:
```bash
# Use smaller batch size for limited GPU memory
bash benchmarks/mathvista.sh --bz 20

# Use larger batch size for high-memory GPUs
bash benchmarks/mathvista.sh --bz 200
```

---

## Monitoring Progress

### Log Files

During execution, logs are stored in:
```
logs/{RUN_ID}/
```

Each benchmark has its own log file:
- `{benchmark}_run.log`: Main execution log from `run_all.sh`
- `{benchmark}_inference.log`: Inference process log
- `{benchmark}_extract.log`: Answer extraction log (if applicable)
- `{benchmark}_judge.log`: Judging/evaluation log (if applicable)

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

After running the benchmarks, results are organized in a unified directory structure:

```
results/{MODEL_IDENTIFIER}/{BENCHMARK_NAME}/{TIMESTAMP}/
```

### Result Locations

All results follow the pattern: `results/{MODEL_IDENTIFIER}/{BENCHMARK_NAME}/{TIMESTAMP}/`

- **MathVista**: `results/{MODEL_ID}/mathvista*/{TIMESTAMP}/mathvista_scores*.json`
- **MathVision**: `results/{MODEL_ID}/mathvision*/{TIMESTAMP}/mathvision_*_judged*.jsonl`
- **MME-RealWorld-Lite**: `results/{MODEL_ID}/mme_realworld*/{TIMESTAMP}/mme_realworld_judged*.jsonl`
- **RealWorldQA**: `results/{MODEL_ID}/realworldqa*/{TIMESTAMP}/realworldqa_judged*.jsonl`
- **BLINK**: `results/{MODEL_ID}/blink*/{TIMESTAMP}/val_results_*.json`
- **ChartQA**: `results/{MODEL_ID}/chartqa*/{TIMESTAMP}/chartqa_*_judged*.jsonl`
- **MMStar**: `results/{MODEL_ID}/mmstar*/{TIMESTAMP}/mmstar_judged*.jsonl`

### Directory Structure Details

- **MODEL_IDENTIFIER**: Automatically generated from checkpoint path (e.g., `trm_140_base-rm_conflict-it1-trm-global_step_80-huggingface`)
- **BENCHMARK_NAME**: Benchmark name with optional CoT suffix (e.g., `mathvista_cot`, `blink`)
- **TIMESTAMP**: Shared timestamp for all benchmarks in the same run (e.g., `20251107_0222`)

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

### GPU Memory Issues

If you encounter GPU out-of-memory errors:
1. Reduce batch size using `--bz` parameter:
   ```bash
   bash benchmarks/mathvista.sh --bz 10  # Use smaller batch size
   ```
2. Check GPU memory usage: `nvidia-smi`
3. Ensure only one benchmark runs per GPU when using `run_all.sh`

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

# Run with custom batch size
bash benchmarks/mathvista.sh --bz 200
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
- ✅ **Automatic GPU allocation** (one GPU per benchmark)
- ✅ **Parallel execution** of all benchmarks
- ✅ **Batch processing** for efficient inference
- ✅ **Unified result directory structure** for easy organization
- ✅ **Flexible selection** of which benchmarks to run
- ✅ **Easy debugging** with comprehensive logging

For help on any script, use `--help` flag:
```bash
bash benchmarks/{benchmark_name}.sh --help
bash run_all.sh --help
```
