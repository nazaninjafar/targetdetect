#define a text classification model to train on the data with pytorch lightning

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AdamW
import torchmetrics



class TextClassificationModel(pl.LightningModule):
    def __init__(self, model, num_labels, lr=2e-5):
        super().__init__()
        self.model = model
      
        self.num_labels = num_labels
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)
        self.f1 = torchmetrics.F1Score(task="multiclass",num_classes=self.num_labels)
        self.precision = torchmetrics.Precision(task="multiclass",num_classes=self.num_labels)
        self.recall = torchmetrics.Recall(task="multiclass",num_classes=self.num_labels)
 
        
        self.save_hyperparameters()


    def forward(self, input_ids, attention_mask, labels=None):
        
        output = self.model(input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1), labels=labels)
        return output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        output = self(input_ids, attention_mask=attention_mask, labels=labels)
  
        loss = output.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        #squeeze input_ids and attention_mask
        input_ids = input_ids
        attention_mask = attention_mask

        output = self(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        logits = output.logits
        preds = torch.argmax(logits, dim=1)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.accuracy(preds, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.f1(preds, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', self.precision(preds, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', self.recall(preds, labels), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        
        return optimizer

    