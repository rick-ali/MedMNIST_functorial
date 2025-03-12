import os
import pytorch_lightning as pl
import argparse
from medmnist import INFO
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datasets.MedMNIST2D_dataset import MedMNISTDataModule
from datasets.CnMedMNIST2D_dataset import CnMedMNISTDataModule
from models.MedMNISTVanilla import MedMNISTModel
from models.FunctorModel import FunctorModel
import torch


def get_dataset_from_args(args):
    if args.dataset == 'pairedcn':
        return CnMedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download, args.x2_angle, args.fixed_covariate)
    if args.dataset == 'vanilla':
        return MedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download)
    raise NotImplementedError


def get_model_from_args(args):
    if args.model == 'vanilla':
        return MedMNISTModel(args.model_flag, n_channels, n_classes, task, args.data_flag, args.size, args.run,
                          milestones=milestones, output_root=log_dir)
    elif args.model == 'functor':
        return FunctorModel(args.model_flag, n_channels, n_classes, task, args.data_flag, args.size, args.run,
                          milestones=milestones, output_root=log_dir, lambda_t=args.lambda_t, lambda_W=args.lambda_W, 
                          latent_transform_process=args.latent_transform_process)
    raise NotImplementedError


def get_args():
    parser = argparse.ArgumentParser(description='Train MedMNIST model with PyTorch Lightning')

    parser.add_argument('--dataset', type=str, default='pairedcn')
    parser.add_argument('--data_flag', type=str, default='pathmnist')
    parser.add_argument('--output_root', type=str, default='tb_logs', help='Where to save logs')
    parser.add_argument('--size', type=int, default=28)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--as_rgb', action='store_true')
    parser.add_argument('--patience', type=int, default=20)
    
    parser.add_argument('--x2_angle', type=float, default=90.0, help='Angle to rotate the second image in paired datasets')
    parser.add_argument('--fixed_covariate', type=int, default=None, help='Fixed covariate for paired datasets')

    
    parser.add_argument('--model_flag', type=str, default='resnet18')
    parser.add_argument('--model', type=str, default='functor')

    parser.add_argument('--lambda_t', type=float, default=0.5)
    parser.add_argument('--lambda_W', type=float, default=0.1)
    parser.add_argument('--latent_transform_process', type=str, default='from_generators')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--run', type=str, default='model1')
    parser.add_argument('--gpu_ids', type=str, default='0')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    args.as_rgb = True

    info = INFO[args.data_flag]
    task = info['task']
    n_channels = 3 if args.as_rgb else info['n_channels']
    n_classes = len(info['label'])

    milestones = [int(0.5 * args.num_epochs), int(0.75 * args.num_epochs)]
    
    ###################################### logger and checkpoints #####################################
    model_name = f'{args.run}_{args.model_flag}_lambdaT_{args.lambda_t}_lambdaW_{args.lambda_W}'
    logger = TensorBoardLogger(
        save_dir=args.output_root,
        name=f'{args.data_flag}/{args.model}/{model_name}'
    )
    log_dir = logger.log_dir

    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    ###################################### data_module #####################################
    data_module = get_dataset_from_args(args)

    ###################################### model #####################################
    model = get_model_from_args(args)
    model.print_hyperparameters()

    ###################################### callbacks #####################################
    early_stop_callback = EarlyStopping(monitor='val_auc', mode='max', patience=args.patience, verbose=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc', 
        mode='max', 
        save_top_k=1,
        dirpath=checkpoints_dir,  # Use the version-specific checkpoints directory
        filename='best_model',
        save_weights_only=True
    )
    
    ###################################### trainer #####################################
    try:
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            default_root_dir=log_dir,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger
        )
        trainer.fit(model, data_module)
    finally:
        best_model_path = checkpoint_callback.best_model_path
        print("Loading model from best checkpoint: ", best_model_path)
        model = type(model).load_from_checkpoint(best_model_path)
        trainer.test(model, data_module)