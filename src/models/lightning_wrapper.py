from typing import Any, Callable, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class LightningWrapper(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_function: Callable=F.cross_entropy, optimizer: torch.optim.Optimizer=torch.optim.Adam, 
                 optimizer_params: Dict[str, Any]={'lr': 0.001}, metrics: Dict[str, torchmetrics.Metric]={}, **pl_module) -> None:
        super(LightningWrapper, self).__init__(**pl_module)
        self.save_hyperparameters()
        self.wrapped_model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.metrics = metrics

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer(self.parameters(), **self.optimizer_params)
    
    def _log_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, step: str) -> None:
        for metric_name, metric_function in self.metrics.items():
            self.log(f'{step}_{metric_name}', metric_function(outputs, targets), prog_bar=True, on_step=False, on_epoch=True)
            metric_function.reset()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrapped_model(x)
        
    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = train_batch
        outputs = self.wrapped_model(inputs)
        loss = self.loss_function(outputs, targets)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        self._log_metrics(outputs, targets, 'train')
        
        return loss

    def validation_step(self, valid_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = valid_batch
        outputs = self.wrapped_model(inputs)
        loss = self.loss_function(outputs, targets)
        self.log('valid_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        self._log_metrics(outputs, targets, 'valid')
        
        return loss
    
    def test_step(self, valid_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = valid_batch
        outputs = self.wrapped_model(inputs)
        loss = self.loss_function(outputs, targets)
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        self._log_metrics(outputs, targets, 'test')

        return loss
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs = batch
        outputs = self.wrapped_model(inputs)
        return torch.softmax(outputs)