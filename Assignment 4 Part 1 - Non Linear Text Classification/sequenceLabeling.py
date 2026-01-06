import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import BertModel


class SequenceLabeling(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the following modules:
            1. Bert Model using the pretrained 'bert-base-uncased' model,
            2. Dropout module.
            3. Linear layer. In dimension should be 768.

        Args:
        num_classes: Number of classes (labels).

        """
        super(SequenceLabeling, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased') # remove None and initialize BERT
        self.dropout = nn.Dropout(p=0.3) # remove None and initialize the Dropout layer
        self.linear = nn.Linear(768, num_classes) # remove None and initialize the Linear layer

    def forward(self, inputs, mask, token_type_ids):
        """
        Implement the forward function to feed the input through the bert model with inputs, mask and token type ids.
        The output of bert layer model is then fed to dropout and the linear layer. 

        Args:
            inputs: Input data. (B, L) tensor of tokens where B is batch size and L is max sequence length.
            mask: attention_mask. (B, L) tensor of binary mask.
            token_type_ids: token type ids. (B, L) tensor

        Returns:
            output: Logits of each label. (B, L, C) tensor of logits where C is number of classes.
        """
        outputs = self.bert(input_ids=inputs, attention_mask=mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  

        x = self.dropout(sequence_output)
        logits = self.linear(x) 

        return logits