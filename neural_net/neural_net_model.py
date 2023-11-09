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
    hidden_layer1_units : int
        number of units for our first hidden layer 
    hidden_layer2_units : int
        number of units for our second hidden layer 
    hidden_layer3_units : int
        number of units for our third hidden layer 
    read_size : int
        number of reads that is gathered from each site

    embedder: nn.Embedding
        embedding layer that helps to embed the 5-mer sequence
    norm0:
        batch norm before passing into the model
    l1:
        first linear layer
    norm1:
        batch norm for the first layer
    relu1:
        first activation function
    dropout1:
        first dropout layer
    l2:
        second linear layer
    norm2:
        batch norm for the second layer
    relu2:
        second activation function
    dropout2:
        second dropout layer
    l3:
        third linear layer
    norm3:
        batch norm for the third layer
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

    def __init__(self, batch_size=128, read_size=20, hidden_layer1_units=320,
                 hidden_layer2_units=160, hidden_layer3_units=64):
        super(NeuralNetModel, self).__init__()
        self.batch_size = batch_size
        self.num_features = 15
        self.num_embedding = 66 # 66 possible 5mers
        self.embedding_dim = 2 # we want to embed into 2 dims
        self.hidden_layer1_units = hidden_layer1_units
        self.hidden_layer2_units = hidden_layer2_units
        self.hidden_layer3_units = hidden_layer3_units
        self.read_size = read_size

        # embedding layer -> embeds each of the 3 5-mers that we have
        self.embedder = nn.Embedding(num_embeddings=self.num_embedding, embedding_dim=self.embedding_dim)
        
        # batch norm before feeding into the network
        self.norm0 = nn.BatchNorm1d(num_features=self.num_features)

        # first Layer (Note that we start off with 15 Features)
        self.l1 = nn.Linear(self.num_features, self.hidden_layer1_units)
        self.norm1 = nn.BatchNorm1d(num_features=self.hidden_layer1_units)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        # second Layer
        self.l2 = nn.Linear(self.hidden_layer1_units, self.hidden_layer2_units)
        self.norm2 = nn.BatchNorm1d(num_features=self.hidden_layer2_units)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)

        # third Layer
        self.l3 = nn.Linear(self.hidden_layer2_units, self.hidden_layer3_units)
        self.norm3 = nn.BatchNorm1d(num_features=self.hidden_layer3_units)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        # fourth Layer
        self.l4 = nn.Linear(self.hidden_layer3_units, 1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):

        # extract the 3 5-mers that are present
        signal_features = x[:, :, :9]
        fivemer_indices = x[:, :, 9:].type(torch.int64)
        # embed the 5-mers and reshape to 6 cols
        embedded_fivemer = self.embedder(fivemer_indices).reshape(-1, self.read_size, 6)

        # concat all the features together
        output = torch.concat([signal_features, embedded_fivemer], axis=2).type(torch.float32)

        # running the first batch norm
        output = output.transpose(dim0=1, dim1=2)
        output = self.norm0(output)
        output = output.transpose(dim0=1, dim1=2)

        # running the first layer
        output = self.l1(output)
        # batch norm across the num channels
        output = output.transpose(dim0=1, dim1=2)
        output = self.norm1(output)
        output = output.transpose(dim0=1, dim1=2)
        output = self.relu1(output)
        output = self.dropout1(output)

        # running the second layer
        output = self.l2(output)
        # batch norm across the num channels
        output = output.transpose(dim0=1, dim1=2)
        output = self.norm2(output)
        output = output.transpose(dim0=1, dim1=2)
        output = self.relu2(output)
        output = self.dropout2(output)

        # running the third layer
        output = self.l3(output)
        # batch norm across the num channels
        output = output.transpose(dim0=1, dim1=2)
        output = self.norm3(output)
        output = output.transpose(dim0=1, dim1=2)
        output = self.relu3(output)
        output = self.dropout3(output)

        # fourth layer and we will pass to a final activation function to get the read label 
        output = self.l4(output)
        output = self.sigmoid(output)
        
        # reshape based on the read size
        output = output.reshape(-1, self.read_size)

        # computing site level probability
        # probability that there is at least one modification from all the reads
        site_prob = 1 - torch.prod(1 - output, axis=1)

        return site_prob 
        