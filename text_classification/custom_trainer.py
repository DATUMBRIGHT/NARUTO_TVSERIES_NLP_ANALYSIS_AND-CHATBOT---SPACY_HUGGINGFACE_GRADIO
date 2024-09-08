from transformers import Trainer
import torch 
from torch.nn import CrossEntropyLoss
from training_utils import get_weights
import pandas as pd
from transformers import AutoModelForSequenceClassification



class CustomTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Ensure you call the parent class's __init__ method
        self.weights = get_weights(self.df)
        self.loss_fcn = CrossEntropyLoss(weight=self.weights)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        self.model.to(self.device)  # Move the model to the appropriate device
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels from inputs
        labels = inputs.pop('labels')
        
        # Move inputs and labels to the appropriate device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute the loss
        loss = self.loss_fcn(logits, labels)
        
        if return_outputs:
            return loss, outputs
        else:
            return loss
        

        
        

    