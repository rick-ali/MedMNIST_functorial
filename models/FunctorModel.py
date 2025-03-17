from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50
from medmnist import Evaluator
from models.BaseModels import ResNet18
from utils.initialise_W_utils import initialise_W_orthogonal, initialise_W_random

class FunctorModel(pl.LightningModule):
    def __init__(self, model_flag, n_channels, n_classes, task, data_flag, size, run,
                 lr=0.001, gamma=0.1, milestones=None, output_root=None, 
                 latent_dim=512, lambda_t=0.5, lambda_W=0.1, modularity_exponent=4,
                 latent_transform_process='from_generators', device='cuda'):
        super().__init__()
        # Save all hyperparameters including new ones for evaluation
        self.save_hyperparameters()

        # Task specifics
        self.task = task
        self.criterion = nn.BCEWithLogitsLoss() if task == "multi-label, binary-class" else nn.CrossEntropyLoss()
        self.output_root = output_root

        # Get Evaluators
        self.train_evaluator = Evaluator(data_flag, 'train', size=size)
        self.val_evaluator = Evaluator(data_flag, 'val', size=size)
        self.test_evaluator = Evaluator(data_flag, 'test', size=size)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Select base model
        if model_flag == 'resnet18':
            self.model = ResNet18(n_channels, n_classes, get_latent=True)
        elif model_flag == 'resnet50':
            self.model = resnet50(pretrained=False, num_classes=n_classes)
        else:
            raise NotImplementedError

        # Initialise functor parameters
        self.lambda_t = lambda_t
        self.lambda_W = lambda_W
        self.latent_dim = latent_dim
        self.modularity_exponent = modularity_exponent
        self.latent_transform_process = latent_transform_process
        if self.latent_transform_process == 'from_generators':
            print("Using latent transformation from generators")
            self.W = nn.Parameter(initialise_W_orthogonal(latent_dim, noise_level=0.3, device=device))
            #self.W = self.identity
        elif self.latent_transform_process == 'decoupled':
            print("Using decoupled latent transformation")
            self.W1 = nn.Parameter(initialise_W_orthogonal(latent_dim, noise_level=0.3, device=device))
            self.W2 = nn.Parameter(initialise_W_orthogonal(latent_dim, noise_level=0.3, device=device))
            self.W3 = nn.Parameter(initialise_W_orthogonal(latent_dim, noise_level=0.3, device=device))
        

    def forward(self, x):
        outputs, latent = self.model(x)
        return outputs, latent
    

    def get_transformed_latent_decoupled(self, latent, transformation_type, covariate):
        latent1 = latent[covariate == 1]
        latent2 = latent[covariate == 2]
        latent3 = latent[covariate == 3]
        transformed_latent1 = F.linear(latent1, self.W1)
        transformed_latent2 = F.linear(latent2, self.W2)
        transformed_latent3 = F.linear(latent3, self.W3)

        transformed_latent = torch.empty_like(latent)
        transformed_latent[covariate == 1] = transformed_latent1
        transformed_latent[covariate == 2] = transformed_latent2
        transformed_latent[covariate == 3] = transformed_latent3

        return transformed_latent

    def get_transformed_latent_from_generator(self, latent, transformation_type, covariate):
        with torch.no_grad():
            W_powers_list = [None] * (covariate.max()+1)
        
        for c in covariate.unique():
            W_powers_list[c] = torch.linalg.matrix_power(self.W, c)

        W_rotation_powers = torch.stack([W_powers_list[c] for c in covariate])
        transformed = torch.bmm(W_rotation_powers, latent.unsqueeze(2)).squeeze(2)

        return transformed
    
    def get_transformed_latent(self, latent, transformation_type, covariate):
        if self.latent_transform_process == 'from_generators':
            transformed = self.get_transformed_latent_from_generator(latent, transformation_type, covariate)
        elif self.latent_transform_process == 'decoupled':
            transformed = self.get_transformed_latent_decoupled(latent, transformation_type, covariate)
        else:
            raise NotImplementedError

        return transformed

    def get_transformation_loss(self, transformed_latent, latent2):
        transformation_loss = nn.functional.mse_loss(transformed_latent, latent2)
        return transformation_loss
    
    def get_modularity_loss(self, W):
        if self.modularity_exponent == 4:
            W_2 = W @ W
            W_4 = W_2 @ W_2
            identity = torch.eye(self.latent_dim, device=W.device)
            modularity_loss = nn.functional.mse_loss(W_4, identity)
        elif self.modularity_exponent == 2:
            W_2 = W @ W
            identity = torch.eye(self.latent_dim, device=W.device)
            modularity_loss = nn.functional.mse_loss(W_2, identity)
        else:
            raise NotImplementedError("Only modularity exponents of 2 and 4 are supported")
        return modularity_loss

    def get_algebra_loss(self):
        if self.latent_transform_process == 'from_generators':
            return self.get_modularity_loss(self.W)
        elif self.latent_transform_process == 'decoupled':
            return self.get_modularity_loss(self.W1) + self.get_modularity_loss(self.W2) + self.get_modularity_loss(self.W3)
        else:
            raise NotImplementedError

    def get_natural_loss(self, outputs, y):
        if self.task == 'multi-label, binary-class':
            targets_proc = y.float()
        else:
            targets_proc = torch.squeeze(y, 1).long()
        loss = self.criterion(outputs, targets_proc)
        return loss

    def calculate_loss(self, batch, batch_idx, stage):
        (x1, y1), (x2, y2), transformation_type, covariate = batch
        labels1 = y1
        labels2 = y1
        
        outputs1, latent1 = self(x1)
        outputs2, latent2 = self(x2)
        

        ########### natural loss ###########
        natural_loss1 = self.get_natural_loss(outputs1, labels1)
        natural_loss2 = self.get_natural_loss(outputs2, labels2)
        natural_loss = 0.5*natural_loss1 + 0.5*natural_loss2


        ########### transformation loss ###########
        if self.lambda_t > 0:
            transformed_latent = self.get_transformed_latent(latent1, transformation_type, covariate)
            transformation_loss = self.get_transformation_loss(transformed_latent, latent2)
        else:
            transformation_loss = 0


        ########### algebra loss ###########
        if self.lambda_t > 0 and self.lambda_W > 0:
            algebra_loss = self.get_algebra_loss()
        else:
            algebra_loss = 0


        ########### logging ###########
        losses = {
            f'{stage}_natural_loss': natural_loss,
            f'{stage}_transformation_loss': transformation_loss,
            f'{stage}_algebra_loss': algebra_loss
        }
        loss = natural_loss + self.lambda_t * transformation_loss + self.lambda_W * algebra_loss
        if stage == 'train':
            losses['loss'] = loss
        else:
            losses[f'{stage}_loss'] = loss
        self.log_dict(losses, prog_bar=True, on_step=False, on_epoch=True)
        

        ########### evaluation ###########
        if stage == 'val':
            self.validation_step_outputs.append(outputs1)
        elif stage == 'test':
            self.test_step_outputs.append(outputs1)
        
        return losses


    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'val')
        return loss

    def on_validation_epoch_end(self):
        result = self.standard_evaluation('val', self.validation_step_outputs, self.val_evaluator)
        self.validation_step_outputs.clear()
        self.log_dict(result)
        return result

    def test_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx, 'test')
        return loss

    def on_test_epoch_end(self):
        result = self.standard_evaluation('test', self.test_step_outputs, self.test_evaluator)
        self.test_step_outputs.clear()
        self.log_dict(result)
        return result
    
    @torch.no_grad()
    def standard_evaluation(self, stage: str, outputs: List[torch.Tensor], evaluator: Evaluator):
        # Skip sanity check
        if self.trainer.sanity_checking:
            self.validation_step_outputs.clear()
            return {f'{stage}_auc': 0, f'{stage}_acc': 0}
        
        logits = torch.cat(outputs, dim=0)
        if self.task == 'multi-label, binary-class':
            y_score = torch.nn.functional.sigmoid(logits)
        else:
            y_score = torch.nn.functional.softmax(logits, dim=1)
        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, self.output_root, self.hparams.run)
        
        return {f'{stage}_auc': auc, f'{stage}_acc': acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
    
    def print_hyperparameters(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Hyperparameter", "Value"]
        
        for key, value in self.hparams.items():
            table.add_row([key, value])
        
        print(table)