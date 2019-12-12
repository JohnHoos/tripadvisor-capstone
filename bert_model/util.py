import numpy as np
import pandas as pd
import nltk.data

import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertForMaskedLM,
    BertTokenizer
)
from transformers.data.processors.utils import InputExample
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler

#################### Global variables ####################

TOKENIZER_PATH = 'data/tokenizers/punkt/english.pickle'


#################### Functions ####################

def loadAndSentencizeData(data_path, tokenizer_path=TOKENIZER_PATH, filter_len=3):
    '''
    Load raw point-of-interest data and return a data frame with an additional column called 'sentencetext'
    Each row contains one review sentence with review-level info

    Parameters
    ----------
    data_path: path
        path to raw data
    tokenizer_path: path
        path to tokenizer (paragraph to sentences)
    filter_len: int, default 3
        sentences with less than this many chars will be dropped

    Return
    ----------
    combinedDF: pandas dataframe
        a pandas dataframe with sentencetext
    '''
    # Load data
    rawData = pd.read_csv(data_path, index_col=0)
    
    print('Raw data loaded')
    # Tokenize and clean
    tokenizer = nltk.data.load(TOKENIZER_PATH)
    sentencesSeries = rawData.reviewtext.apply(lambda x: parse(x, tokenizer))
    print('Reviews divided into sentences')
    index_list = []
    sentences = []

    for index, value in sentencesSeries.items():
        for sent in value:
            index_list.append(index)
            sentences.append(sent.lstrip('.')) # lstrip '.' in each sentence

    print('Generating sentencetext series...')
    tempDF = pd.DataFrame({'sentencetext': sentences, 'original_index': index_list})

    # Combine tempDF with rawData
    combinedDF = tempDF.merge(rawData, left_on='original_index', right_index=True)
    # Filter out sentences whose len <= filter_len (default 3)
    combinedDF['length_char'] = combinedDF.sentencetext.apply(lambda x: len(x))
    combinedDF = combinedDF.loc[combinedDF.length_char > 3].drop(columns=['reviewtext', 'original_index', 'length_char'])
    
    print('Raw dataframe transformed. {} valid review sentences in total after filterng'.format(combinedDF.shape[0]))

    return combinedDF


def parse(text, tokenizer):
    '''
    Parse a single review text
    We observed that the symbol '|' is prevalent in the raw review text and would confuse our tokenizer. Therefore, we replace every '|' with ' .' so that it can be picked up by the tokenizer as a sentence delimiter.
    '''
    # Replace '|' with a white space
    temp = text.replace('|', ' .')
    # Tokenize
    temp = tokenizer.tokenize(temp)
    return temp


def dataframeToDataloader(df, num_classes=4, batch_size=32, max_length=128, is_eval=False):
    '''
    Take in sentencized dataframe and return PyTorch dataloaders

    Parameters
    ----------
    df: pandas dataframe
        sentencized dataframe
    num_classes: int, default 4
        number of target classes. Here we have 'emotion', 'factual', 'tips' and 'other'
    batch_size: int, default 32
        batch size of each batch
    max_length: int, default 128
        max number of tokens in each document
    is_eval: bool, default False
        whether this is val data or train data

    Return
    ----------
    dataloader: pytorch dataloader
        dataloader for training/evaluation
    '''
    # Create a tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    
    print('Converting df to InputExample...')

    # df to InputExample
    examples = [InputExample(guid=i,
                            text_a=text,
                            label=0) for i, text in zip(df.index.tolist(), df.sentencetext.tolist())]

    print('Converting InputExample to Feature...')

    # InputExample to Feature
    features = convert_examples_to_features(examples,
                                        tokenizer,
                                        label_list=list(range(num_classes)), # Since we have 4 classes
                                        max_length=max_length,
                                        output_mode='classification',
                                        pad_on_left=False,
                                        pad_token=tokenizer.pad_token_id,
                                        pad_token_segment_id=0)

    print('Converting Feature to Dataset...')

    # Feature to Dataset
    dataset = TensorDataset(torch.tensor(df.index.tolist(), dtype=torch.long), # this line serves to add a unique identifier for each review sentence
                            torch.tensor([f.input_ids for f in features], dtype=torch.long), 
                            torch.tensor([f.attention_mask for f in features], dtype=torch.long), 
                            torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
                            )

    # torch.tensor([f.label for f in features], dtype=torch.long)

    # Dataset to Dataloader
    if is_eval:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    print('Converting Dataset to Dataloader...')

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return dataloader