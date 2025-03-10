import os
import pytorch_lightning as pl
import argparse
from medmnist import INFO
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datasets.MedMNIST2D_dataset import MedMNISTDataModule
from models.MedMNISTVanilla import MedMNISTModel
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MedMNIST model with PyTorch Lightning')
    parser.add_argument('--data_flag', type=str, default='pathmnist')
    parser.add_argument('--output_root', type=str, default='tb_logs')  # Logs will be saved in tb_logs
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--size', type=int, default=28)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--model_flag', type=str, default='resnet18')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--as_rgb', action='store_true')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--run', type=str, default='model1')
    args = parser.parse_args()

    info = INFO[args.data_flag]
    task = info['task']
    n_channels = 3 if args.as_rgb else info['n_channels']
    n_classes = len(info['label'])

    milestones = [int(0.5 * args.num_epochs), int(0.75 * args.num_epochs)]
    
    # Configure TensorBoard logger with automatic version control
    logger = TensorBoardLogger(
        save_dir=args.output_root,
        name=f'{args.data_flag}/{args.run}'
    )

    # Get the log directory with automatic version
    log_dir = logger.log_dir

    # Create checkpoints directory within the logger's version directory
    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    data_module = MedMNISTDataModule(args.data_flag, args.batch_size, args.resize, args.as_rgb, args.size, args.download)
    model = MedMNISTModel(args.model_flag, n_channels, n_classes, task, args.data_flag, args.size, args.run,
                          milestones=milestones, output_root=log_dir)

    # Set up EarlyStopping and ModelCheckpoint callbacks 
    early_stop_callback = EarlyStopping(monitor='val_auc', mode='max', patience=20, verbose=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc', 
        mode='max', 
        save_top_k=1,
        dirpath=checkpoints_dir,  # Use the version-specific checkpoints directory
        filename='best_model'
    )
    
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        default_root_dir=log_dir,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)