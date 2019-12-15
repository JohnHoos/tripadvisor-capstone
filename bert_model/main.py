'''
Steps
----------
1. Load in and sentencize raw point-of-interest data
2. Convert the sentences into dataloader
3. Load in trained model
4. Generate predictions using the loaded model. For each sentence, provide
    a) probability of each class
    b) predicted label
5. Join the predictions above with original dataframe
6. Save the dataframe in 5
'''

from __future__ import division

import argparse
import glob
import os
import pickle
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange

from util import loadAndSentencizeData, dataframeToDataloader
from model import BERTSequenceClassifier1FC, BERTSequenceClassifier3FC



#################### Define main() ####################

def main():

    ##### Parse args #####

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="The raw point-of-interest data (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the output dataframe with predictions will be saved.")
    # parser.add_argument("--model_type", default=None, type=str, required=True,
    #                     help="The type of model to be loaded. Can be either of ['standard', 'mlm']")    
    parser.add_argument("--model_clf_layers", default=None, type=int, required=True,
                        help="The number of classifying layers. Can be either of [1, 3]")
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="The path to the model")
    parser.add_argument("--config_path", default='config.json', type=str,
                        help="The path to the configuration file")                

    args = parser.parse_args()

    # Sanity check
    # if args.model_type not in ['standard', 'mlm']:
    #     raise ValueError("Invalid input for model_type. Must be either of ['standard', 'mlm']")
    if args.model_clf_layers not in [1, 3]:
        raise ValueError("Invalid input for model_clf_layers. Must be either of [1, 3]")

    ##### Step 1: Load and sentencize data #####
    print('=' * 30, 'Step 1: Load and sentencize data', '=' * 30)
    data = loadAndSentencizeData(args.data_path)

    ##### Step 2: Convert sentences into dataloader #####
    print('=' * 30, 'Step 2: Convert sentences into dataloader', '=' * 30)
    dataloader = dataframeToDataloader(data)

    ##### Step 3: Load in trained model #####
    print('=' * 30, 'Step 3: Load in trained model', '=' * 30)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 

    # Read in configuration file
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    model = None

    # Instantiate model
    if args.model_clf_layers == 1:
        model = BERTSequenceClassifier1FC()
    else:
        model = BERTSequenceClassifier3FC(fc1Size=config['BERT_3']['HIDDEN_1'],
                                          fc2Size=config['BERT_3']['HIDDEN_2'],
                                          dropout_rate=config['BERT_3']['DROPOUT_RATE'])
    # Load weights
    model.load_state_dict(torch.load(args.model_path))

    # if args.model_type == 'standard':
    #     model.loadPretrainedBERT(args.model_path)
    # else:
    #     model.loadFinetunedBERTmlm(args.model_path)

    print('Trained model loaded')

    ##### Step 4: Generate predictions #####
    print('=' * 30, 'Step 4: Generate predictions', '=' * 30)
    
    id_list = []
    pred_list = []
    prob_list_0, prob_list_1, prob_list_2, prob_list_3 = [], [], [], []

    model = model.to(device)

    # For each batch, make and save predictions in lists
    with torch.no_grad():
        for idx, inp, att, typ in tqdm(dataloader):
            idx, inp, att, typ = idx.to(device), inp.to(device), att.to(device), typ.to(device)
            model.eval()
            logits, _ = model(inp, att, typ)
            prob = F.softmax(logits, dim=1)
            id_list.extend(idx.view(-1).tolist())
            pred = prob.max(dim=1)[1]
            pred_list.extend(pred.tolist())
            prob_list_0.extend(prob[:, 0].tolist())
            prob_list_1.extend(prob[:, 1].tolist())
            prob_list_2.extend(prob[:, 2].tolist())
            prob_list_3.extend(prob[:, 3].tolist())
    
    ##### Step 5: Join the predictions #####
    print('=' * 30, 'Step 5: Join the predictions', '=' * 30)
    
    predDF = pd.DataFrame({'prediction': pred_list,
                           'prob_E': prob_list_0,
                           'prob_F': prob_list_1,
                           'prob_O': prob_list_2,
                           'prob_T': prob_list_3}, index=id_list)
    
    predDF['prediction'] = predDF.prediction.astype(str).apply(lambda x: config['LABEL_MAPPING'][x])

    data = data.join(predDF)
    
    ##### Step 6: Save dataframe #####
    print('=' * 30, 'Step 6: Save dataframe', '=' * 30)
    outPath = os.path.join(args.output_dir, 'output_data.csv')
    data.to_csv(outPath)
    print('Output saved to {}'.format(outPath))

if __name__ == '__main__':
    main()
    print('Done')