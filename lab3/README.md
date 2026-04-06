# LAB 3: Sentiment analysis using NLP models

## Prerequisites
The project requires **Python 3**.

#### 1 - Create a Virtual Environment (Optional)
You can use `virtualenv` but we recommend that you use `conda`.
Download the appropriate [Miniconda](https://conda.io/miniconda.html) version for your system. Then follow the installation [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

#### 2 - Install PyTorch
Follow the instructions from the PyTorch home page: https://pytorch.org/

#### 3 - Install Requirements
```
pip install -r lab3/requirements.txt
```

#### 4 - Download pre-trained Word Embeddings

- [Glove Twitter](http://nlp.stanford.edu/data/glove.twitter.27B.zip):  50d vectors
  
The project expects the file(s) to be in the `/embeddings` folder 

A `/datasets` folder is expected to follow the structure in https://github.com/slp-ntua/slp-labs/tree/master/lab3 and contain the same files.

--------------------------------------------------------------------------------------------------------------------
Different versions of main correspond to different models.

`main.py` -> BaselineDNN

`main2.py` -> LSTM

`main3.py` -> SimpleSelfAttentionModel

`main4.py` -> MultiHeadAttentionModel

`main5.py` -> TransformerEncoderModel

There are also options to run pretrained models


`transfer_pretrained.py ` -> get predictions from pre-trained : 
          
`siebert/sentiment-roberta-large-english`, `philipobiorah/bert-imdb-model`, `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
for MR dataset

`cardiffnlp/twitter-roberta-base-sentiment`, `finiteautomata/bertweet-base-sentiment-analysis`, `cardiffnlp/twitter-roberta-base-sentiment-latest`  for Semeval2017A dataset




`finetune_pretrained.py` -> fine tune 'bert-base-cased' model locally




`finetune_pretrained.ipynb` -> fine tuning:

`distilbert-base-uncased`, `albert-base-v2`, `google/electra-small-discriminator` for MR dataset 

`distilroberta-base`, `distilbert-base-uncased`, `cardiffnlp/twitter-roberta-base-sentiment` for Semeval2017A dataset


