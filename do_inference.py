import torch
import time
import argparse
import multiprocessing
import os
import signal
import sys

def occupy_gpu(gpu_id: int, mem_gb: float, util_percent: int):
    """
    在指定的GPU上分配显存并维持一定的计算利用率。
    采用高精度“忙等待”和动态延迟计算以实现最精确的利用率控制。
    """
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)
    
    process_pid = os.getpid()
    print(f"[PID: {process_pid}] [GPU: {gpu_id}] Process started. Target VRAM: {mem_gb:.2f} GB, Target Utilization: {util_percent}%")
    sys.stdout.flush()

    try:
        # 为了包含计算时输出张量的空间，我们将请求的内存分成三份。
        num_elements_per_tensor = int((mem_gb * (1024**3)) / 4 / 3)
        if num_elements_per_tensor <= 0:
            print(f"\n[ERROR] [GPU: {gpu_id}] Memory request ({mem_gb} GB) is too small to create a tensor.")
            return
            
        dim = int(num_elements_per_tensor**0.5)
        if dim == 0:
            print(f"\n[WARNING] [GPU: {gpu_id}] Tensor dimension is zero for {mem_gb} GB. Holding memory without computation.")
            util_percent = 0 # Cannot compute on zero-sized tensor
        
        tensor_a = torch.randn(dim, dim, device=device, dtype=torch.float32)
        tensor_b = torch.randn(dim, dim, device=device, dtype=torch.float32)
        output_tensor = torch.empty(dim, dim, device=device, dtype=torch.float32)
        
        allocated_mem_gb = torch.cuda.memory_allocated(device) / (1024**3)
        print(f"[PID: {process_pid}] [GPU: {gpu_id}] VRAM allocation successful. Actual allocated: {allocated_mem_gb:.2f} GB")
        sys.stdout.flush()

    except Exception as e:
        print(f"\n[ERROR] [GPU: {gpu_id}] An unexpected error occurred during allocation: {e}")
        sys.stdout.flush()
        return

    # --- 利用率控制逻辑 ---
    if util_percent > 0 and dim > 0:
        # --- 校准阶段 (仅用于信息展示) ---
        print(f"[PID: {process_pid}] [GPU: {gpu_id}] Calibrating for stable utilization...")
        sys.stdout.flush()
        calibration_runs = 100
        torch.cuda.synchronize(device)
        start_cal_time = time.perf_counter()
        for _ in range(calibration_runs):
            torch.matmul(tensor_a, tensor_b, out=output_tensor)
        torch.cuda.synchronize(device)
        end_cal_time = time.perf_counter()
        
        avg_op_time = (end_cal_time - start_cal_time) / calibration_runs
        print(f"[PID: {process_pid}] [GPU: {gpu_id}] Calibration complete. Estimated avg op time: {avg_op_time*1e6:.2f} µs.")
        print(f"[PID: {process_pid}] [GPU: {gpu_id}] Starting computation loop with busy-waiting...")
        sys.stdout.flush()

        # --- 主循环 ---
        try:
            while True:
                # 1. 执行一次计算并精确计时
                torch.cuda.synchronize(device)
                op_start_time = time.perf_counter()
                
                torch.matmul(tensor_a, tensor_b, out=output_tensor)
                
                torch.cuda.synchronize(device)
                op_end_time = time.perf_counter()

                # 2. 动态计算本次操作耗时和所需等待时间
                op_duration = op_end_time - op_start_time
                
                # 如果利用率不是100%，则计算等待时间
                if util_percent < 100:
                    wait_duration = op_duration * (100.0 / util_percent - 1.0)
                    
                    # 3. 高精度“忙等待”
                    wait_until = time.perf_counter() + wait_duration
                    while time.perf_counter() < wait_until:
                        pass # 消耗CPU时间以实现精确等待
                        
        except KeyboardInterrupt:
            pass
        finally:
            print(f"\n[PID: {process_pid}] [GPU: {gpu_id}] Exit signal received, cleaning up...")
            sys.stdout.flush()
    else:
        # 如果利用率为0，则只占用显存不计算
        print(f"[PID: {process_pid}] [GPU: {gpu_id}] Utilization is 0, holding VRAM only. Press Ctrl+C to exit.")
        sys.stdout.flush()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            print(f"\n[PID: {process_pid}] [GPU: {gpu_id}] Exit signal received, cleaning up...")
            sys.stdout.flush()


def main():
    """
    主函数，用于解析命令行参数并启动多进程。
    """
    parser = argparse.ArgumentParser(
        description="一个用于占用NVIDIA GPU显存和计算资源的工具 (高精度版)。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
使用示例:
1. 占用 GPU 0 和 1, 每个占用 40GB 显存, 维持 50% 利用率:
   python occupy_gpu.py --gpus 0 1 --mem-gb 40 --utilization 50
"""
    )
    parser.add_argument(
        '--gpus', 
        required=True, 
        type=int, 
        nargs='+', 
        help="要占用的GPU ID列表, 以空格分隔, 例如: --gpus 0 1 2"
    )
    parser.add_argument(
        '--mem-gb', 
        required=True, 
        type=float, 
        help="每个GPU要占用的显存大小 (GB), 例如: --mem-gb 40"
    )
    parser.add_argument(
        '--utilization', 
        required=True, 
        type=int, 
        choices=range(0, 101), 
        metavar="[0-100]",
        help="目标GPU计算利用率 (%%), 0-100之间的整数, 例如: --utilization 80"
    )
    
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA devices found. Please check your NVIDIA driver and PyTorch installation.")
        return

    num_gpus_available = torch.cuda.device_count()
    for gpu_id in args.gpus:
        if gpu_id >= num_gpus_available:
            print(f"[ERROR] GPU ID {gpu_id} is invalid. Available GPU IDs are from 0 to {num_gpus_available - 1}.")
            return

    print("--- GPU Occupancy Tool Started (High-Precision Edition) ---")
    print(f"Target GPUs: {args.gpus}")
    print(f"VRAM per GPU: {args.mem_gb} GB")
    print(f"Target GPU Utilization: {args.utilization}%")
    print("Press Ctrl+C to terminate all processes gracefully.")
    print("-" * 60)
    sys.stdout.flush()

    processes = []
    
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def signal_handler(signum, frame):
        print("\n[Main Process] Termination signal received. Shutting down all child processes...")
        sys.stdout.flush()
        for p in processes:
            if p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        signal.signal(signal.SIGINT, original_sigint_handler)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        for gpu_id in args.gpus:
            p = multiprocessing.Process(target=occupy_gpu, args=(gpu_id, args.mem_gb, args.utilization))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    except Exception as e:
        print(f"[Main Process] An error occurred: {e}")
    finally:
        print("\n--- All tasks have finished. Program exiting. ---")
        sys.stdout.flush()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()