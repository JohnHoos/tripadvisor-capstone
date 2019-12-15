# TripAdvisor Review Text Classification

## Overview
This repository aims to use a text classifier based upon Google's BERT (Bidirectional Encoder Representations from Transformers) algorithm. The repository includes two trained models that one can apply to make prediction. The program expects an input data file in the format of
`tripadvisor_100k_poi_reviews.csv` and carries out the following steps

1. Divide each review entry (a paragraph) into sentences and clean them up.
2. Use the user-specified trained model to generate prediction for each sentence. The prediction includes a predicted label as well as the estimated probability of belonging to each class.
3. Output a data file in the same format as the input file with the aforementioned predictions attached.

## Target Classes
In this project, there are 4 target classes.

1. Emotional (E)
2. Factual (F)
3. Tips (T)
4. Other (O)

In the final output, predicted class will be denoted by the letter between the parenthesis.

## Run

### Step 1
In order to run predictions using a trained model, first go to `run.sh`. In `run.sh`, there is only one line of command 
```
python main.py --data_path data/tripadvisor_100k_poi_reviews.csv --output_dir output --model_clf_layers 3 --model_path model/3_layer_mlm.pt
```
Please alter the arguments based upon your need. For instance, you might want to specify which model to use by changing `--model_clf_layers   ` and `--model_path`. (For model-related arguments, please view the **Models** section)

### Step 2
Once you alter `run.sh`, execute the following command in terminal.

```
sh run.sh
```

### Step 3
The program will start running and the output will be saved in your specified directory.

## Models
This repository contains 2 trained models. All model files should be placed in the `model` directory.

### Model 1 (defualt)
**file name**: `3_layer_mlm.pt`

**num_clf_layers**: 3

**training scheme**: First trained on unlabeled reviews using Masked Language Modeling (unsupervised learning). Then trained on labeled data of around 3,200 instances (supervised learning), with BERT weights frozen.

### Model 2
**file name**: `1_layer_unfrozeon.pt`

**num_clf_layers**: 1

**training scheme**: Only trained on labeled data of around 3,200 instances (supervised learning), with BERT weights unfrozen.