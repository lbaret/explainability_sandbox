from pathlib import Path
from tabnanny import check

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights

from src.models.lightning_wrapper import LightningWrapper
from src.models.resnet50 import resnet50


@click.command()
@click.option('-d', '--data-root-folder', type=str, required=True, help='Root folder to load fruits vegetables 360 dataset')
@click.option('-c', '--checkpoints-path', type=str, required=True, help='PyTorch Lightning checkpoints path')
@click.option('-r', '--train-ratio', type=float, default=0.9, help='Train ratio for training set splitting into train/valid sets')
@click.option('-b', '--batch_size', type=int, default=512, help='Size of batch for model finetuning and testing')
@click.option('-w', '--num_workers', type=int, default=2, help="Num workers for data loader parallelization handling")
@click.option('-e', '--epochs', type=int, default=25, help="Number of epochs for training step")
def finetune_resnet(data_root_folder: str, checkpoints_path: str, train_ratio: float, batch_size: int, num_workers: int, epochs: int):
    data_root_folder_object = Path(data_root_folder)
    checkpoints_path_object = Path(checkpoints_path)
    
    train_set = ImageFolder(data_root_folder_object.joinpath('Training'), transform=transforms.ToTensor())
    test_set = ImageFolder(data_root_folder_object.joinpath('Test'), transform=transforms.ToTensor())

    # Split training set
    total_size = len(train_set)
    train_size = int(train_ratio * total_size)
    valid_size = total_size - train_size

    train_set, valid_set = random_split(train_set, [train_size, valid_size])

    # Loading model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Changing classifier output dimension
    model.fc = nn.Linear(in_features=2048, out_features=131)

    # Freeze unsollicited weights
    for name, params in model.named_parameters():
        if 'fc' in name:
            continue
        params.requires_grad = False

    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_path_object,
        save_last=True,
        every_n_epochs=1
    )

    # Model and trainer
    lightning_model = LightningWrapper(model)

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(accelerator=device, max_epochs=epochs, callbacks=[model_checkpoint])

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    # Training
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # Testing
    trainer.test(lightning_model, dataloaders=test_loader)

if __name__ == '__main__':
    finetune_resnet()