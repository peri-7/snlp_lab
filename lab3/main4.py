import os
import warnings
import matplotlib.pyplot as plt
import numpy as np

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from attention import MultiHeadAttentionModel
from early_stopper import EarlyStopper
from training import train_dataset, eval_dataset, get_metrics_report, torch_train_val_split
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B/glove.twitter.27B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50

DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"
#DATASET = "MR"


# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()
le.fit(y_train)
y_train_og = y_train
y_train = le.transform(y_train) # EX1
y_test = le.transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size

print('\n ========== EX1 ========== \n')
for i in range(10):
    print(f'{y_train[i]}, label:{y_train_og[i]}')


# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

print('\n ========== EX2 ========== \n')
for i in range(10):
    print(f'tweet {i+1}:\n{train_set.data[i]}\n')

print('\n ========== EX3 ========== \n')
for i in range(10, 15):
    sentence, label, length = train_set[i]
    print(f'tweet {i+1}:\n{train_set.data[i]}\ntweet {i+1} preprocessed for neural network: {sentence}\n label: {label} \n length = {length}\n')

# EX7 - Define our PyTorch-based DataLoader
train_loader, val_loader = torch_train_val_split(train_set, batch_train=BATCH_SIZE, batch_eval=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
out_dim = 1 if n_classes == 2 else n_classes

model = MultiHeadAttentionModel(
                    output_size=out_dim,  # EX8
                    embeddings=embeddings,
                    n_head=5
                    )

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
if n_classes == 2:
    criterion = torch.nn.BCEWithLogitsLoss() # EX8
else:
    criterion = torch.nn.CrossEntropyLoss() # EX8
parameters = [p for p in model.parameters() if p.requires_grad]# EX8
optimizer = torch.optim.Adam(params = parameters, lr=0.001)  # EX8

#############################################################################
# Training Pipeline
#############################################################################
train_losses = []
val_losses = []
test_losses = []
best_path = "multi_head_attention_best_model.pt"
stopper = EarlyStopper(model, save_path=best_path, patience=5, min_delta=0)
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)
                                                            
    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader,
                                                            model,
                                                            criterion)                                                       

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
        
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)
    
    if stopper.early_stop(val_loss):
        print(f'stopped at epoch {epoch}')
        break

print('\n ========== EX10 ========== \n')

model.load_state_dict(torch.load(best_path))
_, (best_y_test_gold, best_y_test_pred) = eval_dataset(test_loader, model, criterion)

print(get_metrics_report(best_y_test_gold, best_y_test_pred))

plt.figure()
x = np.arange(len(train_losses))
plt.plot(x, train_losses, label="Train")
plt.plot(x, val_losses, label="Validation")
plt.plot(x, test_losses, label="Test")
plt.title(f"MultiHeadAttentionModel - Learning Curves ({DATASET})")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()
