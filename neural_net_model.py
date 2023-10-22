import torch
import torch.nn as nn
import numpy as np

class NeuralNetModel(nn.Module):
    """
    Neural Network Model for identification of M6A Modifications

    ...

    Attributes
    ----------
    batch_size : int
        batch_size that was used for the `DataLoader` object
    hidden_layers : int
        number of hidden layer units we want for our hidden layer
    read_size : int
        number of reads that is gathered from each site

    embedder: nn.Embedding
        embedding layer that helps to embed the 5-mer sequence
    l1:
        first linear layer
    relu1:
        first activation function
    dropout1:
        first dropout layer
    l2:
        second linear layer
    relu2:
        second activation function
    dropout2:
        second dropout layer
    l3:
        third linear layer
    relu3:
        third activation function
    dropout3:
        third dropout layer
    l4:
        fourth linear layer
    sigmoid:
        sigmoid layer to get the final probability for each read


    Methods
    -------
    
    """

    def __init__(self, batch_size=128, read_size=20, hidden_layers=256):
        super(NeuralNetModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.read_size = read_size

        # embedding layer -> embeds each of the 3 5-mers that we have
        self.embedder = nn.Embedding(num_embeddings=66, embedding_dim=2)

        # first Layer (Note that we start off with 15 Features)
        self.l1 = nn.Linear(15, self.hidden_layers)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        # second Layer
        self.l2 = nn.Linear(self.hidden_layers, 150)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        # third Layer
        self.l3 = nn.Linear(150, 32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)

        # fourth Layer
        self.l4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):

        # extract the 3 5-mers that are present and embed the vector
        signal_features = x[:, :, :9]
        fivemer_indices = x[:, :, 9:].type(torch.int64)
        # embed the 5-mers and reshape to 6 cols
        embedded_fivemer = self.embedder(fivemer_indices).reshape(-1, self.read_size, 6)

        # concat all the features together
        new_x = torch.concat([signal_features, embedded_fivemer], axis=2).type(torch.float32)

        # running the first layer
        output = self.l1(new_x)
        output = self.relu1(output)
        output = self.dropout1(output)

        # running the second layer
        output = self.l2(output)
        output = self.relu2(output)
        output = self.dropout2(output)

        # running the third layer
        output = self.l3(output)
        output = self.relu3(output)
        output = self.dropout3(output)

        # fourth layer and we will pass to a final activation function to get the read label 
        output = self.l4(output)
        output = self.sigmoid(output)
        output = output.reshape(-1, self.read_size)

        # computing site level probability
        # probability that there is at least one modification from all the reads
        site_prob = 1 - torch.prod(1 - output, axis=1)

        return site_prob 
        