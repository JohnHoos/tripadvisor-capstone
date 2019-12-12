#!/bin/bash

python main.py --data_path data/tripadvisor_100k_poi_reviews.csv --output_dir output --model_clf_layers 3 --model_path model/3_layer_mlm_unfrozen.pt

