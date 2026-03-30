import torch
import numpy as np
from torch import nn
from torch.nn import init


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        ...  # EX4

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        ...  # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        ...  # EX4
        
        self.emb = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings), freeze = not trainable_emb, padding_idx=0) 


        # 4 - define a non-linear transformation of the representations
        ...  # EX5
        
        vocab, emb_dim = embeddings.shape
        hidden_dim = 128
        
        #self.linear1 = nn.Linear(emb_dim, hidden_dim)
        self.linear1 = nn.Linear(2*emb_dim, hidden_dim)
        init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        init.zeros_(self.linear1.bias)
        self.activ1 = nn.ReLU()
        
        
        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5
        self.linear2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        
        # 1 - embed the words, using the embedding layer
        embeddings = self.emb(x)# EX6 (B, N_max, D_emb)
        
        
        # 2 - construct a sentence representation out of the word embeddings
        '''
        useful for comparison
        sum_embeddings = torch.sum(embeddings, dim = 1)
        lengths_tensor = lengths.view(-1, 1).float()
        representations = sum_embeddings / lengths_tensor
        '''
        # Q1
        sum_embeddings = torch.sum(embeddings, dim = 1)
        lengths_tensor = lengths.view(-1, 1).float()
        avg_embeddings = sum_embeddings / lengths_tensor
        
        mask_max = (x == 0).unsqueeze(2) #(B, Nmax, 1)
        max_embeddings = embeddings.masked_fill(mask_max, -float('inf'))
        max_embeddings = torch.max(max_embeddings, dim = 1).values
        # no need to rule out the zero indexes, already done in embedding layer
        representations = torch.cat((avg_embeddings, max_embeddings), dim=1)
        
            

        # 3 - transform the representations to new ones.
        
        representations = self.linear1(representations) 
        representations = self.activ1(representations)
  
        
        # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.linear2(representations)  # EX6

        return logits


class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)
        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)

        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)
        batch_indices = torch.arange(batch_size, device=x.device)
        representations = ht[batch_indices, lengths - 1, :]

        logits = self.linear(representations)

        return logits
