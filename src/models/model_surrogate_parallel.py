from typing import Any, Callable, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .deep_linear import DeepLinear
from .simple_logistic_surrogate import SimpleLogisticSurrogate


class ModelSurrogateParallel(pl.LightningModule):
    def __init__(self, in_features: int, out_features: int, loss_function: Callable=F.cross_entropy, optimizer: torch.optim.Optimizer=torch.optim.Adam, 
               optimizer_params: Dict[str, Any]={'lr': 0.001}, **pl_module) -> None:
        super(ModelSurrogateParallel, self).__init__(**pl_module)

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.save_hyperparameters()

        self.deep_linear = DeepLinear(in_features, out_features)
        self.surrogate = SimpleLogisticSurrogate(in_features, out_features)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer(self.parameters(), **self.optimizer_params)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_model = self.deep_linear(x)
        y_surrogate = self.surrogate(x)
        return y_model, y_surrogate
    
    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = train_batch
        scores_model, scores_surrogate = self.forward(inputs)

        targets_model = torch.argmax(torch.softmax(scores_model, 1), 1).detach()

        loss_model = self.loss_function(scores_model, targets)
        loss_surrogate = self.loss_function(scores_surrogate, targets_model)

        self.log('train_loss_model', loss_model)
        self.log('train_loss_surrogate', loss_surrogate)
        
        return loss_model + loss_surrogate
    
    def validation_step(self, valid_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = valid_batch
        scores_model, scores_surrogate = self.forward(inputs)

        targets_model = torch.argmax(torch.softmax(scores_model, 1), 1).detach()

        loss_model = self.loss_function(scores_model, targets)
        loss_surrogate = self.loss_function(scores_surrogate, targets_model)

        self.log('valid_loss_model', loss_model)
        self.log('valid_loss_surrogate', loss_surrogate)
        
        return loss_model + loss_surrogate
    
    def test_step(self, test_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = test_batch
        scores_model, scores_surrogate = self.forward(inputs)

        targets_model = torch.argmax(torch.softmax(scores_model, 1), 1).detach()

        loss_model = self.loss_function(scores_model, targets)
        loss_surrogate = self.loss_function(scores_surrogate, targets_model)

        self.log('test_loss_model', loss_model)
        self.log('test_loss_surrogate', loss_surrogate)
        
        return loss_model + loss_surrogate
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(batch) == 2:
            inputs, _ = batch
        else:
            inputs = batch
        scores_model, scores_surrogate = self.forward(inputs)

        y_model = torch.argmax(torch.softmax(scores_model, 1), 1)
        y_surrogate = torch.argmax(torch.softmax(scores_surrogate, 1), 1)

        return y_model, y_surrogate