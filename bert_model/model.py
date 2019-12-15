import copy
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertForMaskedLM,
    BertTokenizer
)


########## Model Architectures ##########

class BERTSequenceClassifier1FC(nn.Module):
    '''
    A BERT classifier with 1 fully connected layer
    '''
    def __init__(self, num_classes=4, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        # Freeze all weights in bert if freeze_bert is True
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.num_classes = num_classes
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               token_type_ids=token_type_ids)
        h_cls = h[:, 0] # 0 means take h_cls only
        logits = self.classifier(h_cls)
        return logits, attn

    def loadPretrainedBERT(self, path, freeze=True):
        self.bert.load_state_dict(torch.load(path))
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        print('Pre-trained BERT weights loaded')

    def loadFinetunedBERTmlm(self, path, freeze=True):
        temp = BertForMaskedLM.from_pretrained(path, output_attentions=True)
        self.bert = copy.deepcopy(temp.bert)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        print('Fine-tuned MLM BERT weights loaded')


class BERTSequenceClassifier3FC(nn.Module):
    '''
    A BERT classifier with 3 fully connected layers
    '''
    def __init__(self, num_classes=4, fc1Size=256, fc2Size=64, dropout_rate=0.1, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        # Freeze all weights in bert if freeze_bert is True
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size, fc1Size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(dropout_rate), # Dropout
                                        nn.Linear(fc1Size, fc2Size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(fc2Size, num_classes)
                                        )
        self.num_classes = num_classes
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               token_type_ids=token_type_ids)
        h_cls = h[:, 0] # 0 means take h_cls only
        logits = self.classifier(h_cls)
        return logits, attn

    def loadPretrainedBERT(self, path, freeze=True):
        self.bert.load_state_dict(torch.load(path))
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        print('Pre-trained BERT weights loaded')

    def loadFinetunedBERTmlm(self, path, freeze=True):
        temp = BertForMaskedLM.from_pretrained(path, output_attentions=True)
        self.bert = copy.deepcopy(temp.bert)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        print('Fine-tuned MLM BERT weights loaded')