import os
import wandb
import pytorch_lightning as pl
import argparse
from medmnist import INFO
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasets.MedMNIST2D_dataset import MedMNISTDataModule
from datasets.CnMedMNIST2D_dataset import CnMedMNISTDataModule
from datasets.Z2MedMNIST2D_dataset import Z2MedMNISTDataModule
from models.MedMNISTVanilla import MedMNISTModel
from models.FunctorModel import FunctorModel
import torch

def get_dataset_from_args(args):
    if args.dataset == 'pairedcn':
        return CnMedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download, args.x2_angle, args.fixed_covariate)
    if args.dataset == 'vanilla':
        return MedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download)
    if args.dataset == 'z2':
        return Z2MedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download)
    raise NotImplementedError

def get_model_from_args(args):
    info = INFO[args.data_flag]
    task = info['task']
    n_channels = 3 if args.as_rgb else info['n_channels']
    n_classes = len(info['label'])

    milestones = [int(0.5 * args.num_epochs), int(0.75 * args.num_epochs)]
    if args.model == 'vanilla':
        return MedMNISTModel(args.model_flag, n_channels, n_classes, task, args.data_flag, args.size, args.run,
                          milestones=milestones)
    
    elif args.model == 'functor':
        return FunctorModel(args.model_flag, n_channels, n_classes, task, args.data_flag, args.size, args.run, device=args.device,
                          milestones=milestones, lambda_c=args.lambda_c,
                          lambda_t=args.lambda_t, lambda_W=args.lambda_W, algebra_loss_criterion=args.algebra_loss_criterion,
                          W_init=args.W_init, fix_rep=args.fix_rep, W_block_size=args.block_size,
                          latent_transform_process=args.latent_transform_process, modularity_exponent=args.modularity_exponent,
                          lr=args.lr, gamma=args.gamma)
    
    raise NotImplementedError

def train():
    """Runs a single training experiment with W&B hyperparameter tuning."""
    wandb.init()
    args = get_args()
    
    # Override args with sweep config
    args.lambda_c = wandb.config.lambda_c
    args.lambda_t = wandb.config.lambda_t
    args.lambda_W = wandb.config.lambda_W
    

    if args.gpu_id != '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    args.as_rgb = True if not args.not_rgb else False

    # Wandb logger
    wandb_logger = WandbLogger(project="MedMNIST-HP-Search", name=f"{args.model}_lr{args.lr}_lambdaT{args.lambda_t}")
    
    # Data Module
    if args.dataset == 'pairedcn':
        args.modularity_exponent = 4
    elif args.dataset == 'z2':
        args.modularity_exponent = 2
    data_module = get_dataset_from_args(args)
    model = get_model_from_args(args).to(device)

    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best_model"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[int(args.gpu_id)],
        logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    trainer.fit(model, data_module)

    # Log best validation loss
    best_model_path = checkpoint_callback.best_model_path
    model = type(model).load_from_checkpoint(best_model_path)
    test_results = trainer.test(model, data_module)[0]

    # Log metrics to W&B
    wandb.log(test_results)

def get_args():
    parser = argparse.ArgumentParser(description="Train MedMNIST model with PyTorch Lightning")
    
    parser.add_argument('--dataset', type=str, default='z2')
    parser.add_argument('--data_flag', type=str, default='pathmnist')
    parser.add_argument('--output_root', type=str, default='tb_logs')
    parser.add_argument('--size', type=int, default=28)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--not_rgb', action='store_true')
    parser.add_argument('--patience', type=int, default=20)
    
    parser.add_argument('--x2_angle', type=float, default=90.0)
    parser.add_argument('--fixed_covariate', type=int, default=None)

    parser.add_argument('--model_flag', type=str, default='resnet18')
    parser.add_argument('--model', type=str, default='functor')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--lambda_c', type=float, default=1.0)
    parser.add_argument('--lambda_t', type=float, default=0.5)
    parser.add_argument('--lambda_W', type=float, default=0.5)
    parser.add_argument('--latent_transform_process', type=str, default='from_generators')
    parser.add_argument('--W_init', type=str, default='orthogonal')
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--fix_rep', action='store_true')
    parser.add_argument('--algebra_loss_criterion', type=str, default='mse')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--run', type=str, default='model1')
    parser.add_argument('--visible_gpus', type=str, default='0,1,2,3')
    parser.add_argument('--gpu_id', type=str, default='0')

    return parser.parse_args()

if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_unreg_loss", "goal": "minimize"},
        "parameters": {
            #"lr": {"values": [0.001, 0.0005, 0.0001]},
            #"batch_size": {"values": [64, 128, 256]},
            "lambda_c": {"values": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
            "lambda_t": {"values": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
            "lambda_W": {"values": [0.0]},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="MedMNIST-HP-Search")
    wandb.agent(sweep_id, function=train, count=20, project="MedMNIST-HP-Search")
