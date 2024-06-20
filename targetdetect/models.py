#define a text classification model to train on the data with pytorch lightning

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AdamW
import torchmetrics

from llm_util import get_llm_engine
import tqdm
import jsonlines
import openai
# from openai import OpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  



openai.api_key=""

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)
class GPTInference:
    def __init__(self,llm_model_name):
        super().__init__()
        self.model_engine = llm_model_name

        # if self.llm_model_name =='gpt35':
        #     self.model_engine = "gpt-3.5-turbo"
        # elif self.llm_model_name =='gpt4':
        #     self.model_engine = "gpt-4"

    def get_model_predict(self,prompt):
        
        completion = completion_with_backoff(model = self.model_engine,
                    messages = [{"role": "user", "content": prompt}],
                    n=1,stop=None,temperature=0)
        
        return completion.choices[0].message.content

class OpenLLMInference:
    def __init__(self,llm_model_name):
        super().__init__()
        
        # self.model_id = "meta-llama/Llama-2-13b-chat-hf"
        self.model_id = llm_model_name 
        self.llm, self.sampling_params = get_llm_engine(self.model_id)
        

    def get_model_predict(self,prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)

        return [output_text.outputs[0].text for output_text in outputs]



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

    