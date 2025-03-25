import subprocess
from medmnist import INFO
import argparse
import time
from joblib import Parallel, delayed


def run_experiments_on_gpu(n_runs, gpu_id, chunk_size):    
    time.sleep(int(gpu_id) * 5)
    base_commands = [
        'python3 lightning_train_and_eval.py --run=vanilla_augmented ' + \
        f'--lambda_t=0.0 --lambda_W=0.0 --data_flag=pathmnist --dataset=z2 --gpu_id={gpu_id}',
    ]

    commands = base_commands * n_runs

    # divide commands into chunks of size 4
    chunks = [commands[i:i+chunk_size] for i in range(0, len(commands), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        processes = []
        for command in chunk:
            print(command)
            processes.append(subprocess.Popen(command, shell=True))
            time.sleep(5)
        for process in processes: process.wait()


if __name__ == '__main__':
    N_EXPERIMENTS_PER_GPU = 4
    N_PARALLEL_EXPERIMENTS_PER_GPU = 2
    GPUS = ['0', '1', '2', '3']

    print(f"Running a total of {N_EXPERIMENTS_PER_GPU * len(GPUS)} experiments on {len(GPUS)} GPUs")
    time.sleep(2)

    # Run experiments in parallel on all GPUs
    Parallel(n_jobs=len(GPUS))(delayed(run_experiments_on_gpu)(N_EXPERIMENTS_PER_GPU, gpu_id, N_PARALLEL_EXPERIMENTS_PER_GPU) for gpu_id in GPUS)
        
