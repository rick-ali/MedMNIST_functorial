import subprocess

if __name__ == "__main__":
    N = 5
    commands = []

    terminal_commands =[
        "python3 lightning_train_and_eval.py --run=VANILLA --lambda_t=0. --lambda_W=0",
        "python3 lightning_train_and_eval.py --run=FROM_GENERATORS --lambda_t=0.1 --lambda_W=0",
        "python3 lightning_train_and_eval.py --run=FROM_GENERATORS --lambda_t=0.5 --lambda_W=0",
        "python3 lightning_train_and_eval.py --run=FROM_GENERATORS --lambda_t=0.5 --lambda_W=0.1",
        "python3 lightning_train_and_eval.py --run=DECOUPLED --lambda_t=0.5 --lambda_W=0",
        "python3 lightning_train_and_eval.py --run=DECOUPLED --lambda_t=0.5 --lambda_W=0.1",
        "python3 lightning_train_and_eval.py --run=FROM_GENERATORS_1_COVARIATE --lambda_t=0.5 --lambda_W=0 --fixed_covariate=1",
        "python3 lightning_train_and_eval.py --run=FROM_GENERATORS_1_COVARIATE --lambda_t=0.5 --lambda_W=0.1 --fixed_covariate=1",
    ]

    # Repeat each experiment N times
    for _ in range(N):
        for i in range(int(len(terminal_commands)/2)):
            commands = terminal_commands[2*i: 2*i+2]

            processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]
            for p in processes:
                p.wait()
            commands = []

        
    
