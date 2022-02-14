# tripadvisor-capstone

Multiclass topic classification (factual/emotional/tips/others) on user reviews. This GitHub repo does not contain the dataset used for training and testing as they are propritetary information, but the fully trained BERT model is available. A poster of the project can be found [here](https://app.box.com/s/e31d615oh9f04ugmipzqx9c8s8flwdlb).

### Data:
- 100k of unlabeled user reviews
- labeled data as 'facts' and 'tips'

### Model:
- Bert (best at 77% test accuracy)
- Conventional ML models (fully superpervised)
- LDA / Guided LDA (unsupervised and semi-supervised)
