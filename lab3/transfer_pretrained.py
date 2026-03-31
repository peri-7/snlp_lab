from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report


DATASET = 'MR'
PRETRAINED_MODEL = ['siebert/sentiment-roberta-large-english',
                    'philipobiorah/bert-imdb-model',
                    'distilbert/distilbert-base-uncased-finetuned-sst-2-english']
'''
DATASET = 'Semeval2017A'
PRETRAINED_MODEL = ['cardiffnlp/twitter-roberta-base-sentiment', 
                    'finiteautomata/bertweet-base-sentiment-analysis',
                    'cardiffnlp/twitter-roberta-base-sentiment-latest']    
'''


LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'philipobiorah/bert-imdb-model': {
        'LABEL_0': 'negative',
        'LABEL_1': 'positive',
    },
    'distilbert/distilbert-base-uncased-finetuned-sst-2-english': {
        'NEGATIVE': 'negative',
        'POSITIVE': 'positive',
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'cardiffnlp/twitter-roberta-base-sentiment-latest': {
        'negative': 'negative',
        'neutral': 'neutral',
        'positive': 'positive',
    },
    'finiteautomata/bertweet-base-sentiment-analysis': {
        'NEG': 'negative',
        'NEU': 'neutral',
        'POS': 'positive',
    },
}

if __name__ == '__main__':
    # load the raw data
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    n_classes = len(list(le.classes_))

    # define a proper pipeline
    sentiment_pipeline = {m: pipeline("sentiment-analysis", model=m, truncation=True) for m in PRETRAINED_MODEL }

    
    for m in PRETRAINED_MODEL:
        y_pred = []
        model = sentiment_pipeline[m]
        for x in tqdm(X_test):
            # TODO: Main-lab-Q6 - get the label using the defined pipeline 
            label = model(x)[0]['label']
            y_pred.append(LABELS_MAPPING[m][label])

        y_pred = le.transform(y_pred)
        print(f'\nDataset: {DATASET}\nPre-Trained model: {m}\nTest set evaluation\n{get_metrics_report([y_test], [y_pred])}')
