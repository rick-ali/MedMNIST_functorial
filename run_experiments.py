import subprocess
from medmnist import INFO


if __name__ == "__main__":
    N = 5
    commands = []
    datasets = list(INFO.keys())

    # terminal_commands =[
    #     "python3 lightning_train_and_eval.py --run=VANILLA --lambda_t=0. --lambda_W=0",
    #     "python3 lightning_train_and_eval.py --run=FROM_GENERATORS --lambda_t=0.1 --lambda_W=0",
    #     "python3 lightning_train_and_eval.py --run=FROM_GENERATORS --lambda_t=0.5 --lambda_W=0",
    #     "python3 lightning_train_and_eval.py --run=FROM_GENERATORS --lambda_t=0.5 --lambda_W=0.1",
    #     "python3 lightning_train_and_eval.py --run=DECOUPLED --lambda_t=0.5 --lambda_W=0",
    #     "python3 lightning_train_and_eval.py --run=DECOUPLED --lambda_t=0.5 --lambda_W=0.1",
    #     "python3 lightning_train_and_eval.py --run=FROM_GENERATORS_1_COVARIATE --lambda_t=0.5 --lambda_W=0 --fixed_covariate=1",
    #     "python3 lightning_train_and_eval.py --run=FROM_GENERATORS_1_COVARIATE --lambda_t=0.5 --lambda_W=0.1 --fixed_covariate=1",
    # ]
    terminal_commands = []
    for dataset in datasets:
        for coefficient in [0.0, 0.1]:
            run_name = f"VANILLA_AUGMENTED" if coefficient == 0.0 else f"FROM_GENERATORS_1_COVARIATE_MSE"
            cmd = f"python3 lightning_train_and_eval.py --run={run_name} --lambda_t={coefficient} --lambda_W={coefficient} --fixed_covariate=1 --data_flag={dataset} --download"
            terminal_commands.append(cmd)

    # Repeat each experiment N times
    for _ in range(N):
        for i in range(int(len(terminal_commands)/3)):
            commands = terminal_commands[3*i: 3*i+3]
            print("processing commands: ", commands)
            processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]
            for p in processes:
                p.wait()
            commands = []

        
    
