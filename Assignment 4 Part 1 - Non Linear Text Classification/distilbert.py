import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import DistilBertModel


class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the following modules:
            1. DistilBert Model using the pretrained 'distilbert-base-uncased' model
            2. Linear layer
            3. Any other layers to help with accuracy

        Args:
            num_classes: Number of classes (labels).

        """
        super(DistilBERTClassifier, self).__init__()

        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')# remove None and initialize DistilBERT
        self.dropout = nn.Dropout(0.3) # remove None and initialize the Dropout layer
        self.linear = nn.Linear(768, num_classes) # remove None and initialize the Linear layer

    def forward(self, inputs, mask):
        """
        Implement the forward function to feed the input through the distilbert model with inputs and mask.
        Use the DistilBert output to obtain logits of each label. 
        Ref: https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertModel

        Args:
            inputs: Input data. (B, L) tensor of tokens where B is batch size and L is max sequence length.
            mask: attention_mask. (B, L) tensor of binary mask.

        Returns:
            output: Logits of each label. (B, C) tensor of logits where C is number of classes.
        """
        outputs = self.distilbert(input_ids=inputs, attention_mask=mask)
        cls_output = outputs.last_hidden_state[:, 0] 
        cls_output = self.dropout(cls_output)
        logits = self.linear(cls_output)
        return logits
