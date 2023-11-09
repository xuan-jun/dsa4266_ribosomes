# torch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# libraries for preprocessing
from itertools import product
import numpy as np
import pandas as pd
import gzip
import json
import random
import os

class RNAData(Dataset):
    """
    Dataset object

    Attributes
    ----------
    data_path : str
        full path to the .json file that contain the data required for training/testing
    train : bool
        indicates whether we are using this dataset for training or just purely testing
    sgNex: bool
        indicates whether we are using sgNex data. This allows for check if the file is a .json
    read_size : int
        number of reads that we will be using for each site (transcript_id, position)
    batch_size : int
        number of sites per read that we will put into the Neural Network per batch
    data_dict : Dict
        contains all the data for each site
        Key: (transcript_id, position)
        Value [Dict]:  Contains the following keys
            'kmer_sequence' [str] - kmer sequence of length 7 at the site
            'signal_features' [list] - 2D array of size (n x 9), where n is the number of reads for that site that is passed in
            'label' [bool] - Label for the site (if train=True)
    fivemer_mapping : Dict
        contains the mapping of each 5-mer sequence to its corresponding integer representation
    kmer_sequence_mapping: Dict
        contains a 3 index sequence for each possible site that we can have
        Key: (transcript_id, position)
        Value [list]: Length 3 tuple (left_5mer, center_5mer, right_5mer) of their corresponding fivemer mapping index
    length: int
        length of the training set (if train=True and eval=False), length of the eval set (if train=True and eval=True),
        lenght of whole set (if train=False)
    train_size (optional): int
        size of the training set (only applicable if train=True)
    label_path (optional): str
        full path to the .info file that contains the labels for the training data (only applicable if train=True)
    gene_dict (optional): Dict
        contains the various sites that are under a particular gene_id (only applicable if train=True)
        Key: gene_id
        Value: List of the sites (transcript_id, position) 
    train_transcript_position (optional): List
        contains the sites that are under the training set (only applicable if train=True)
    test_transcript_position (optional): List
        contains the sites that are under the testing set (only applicable if train=True)
    eval (optional): bool
        indicator of whether we are looking at the training set data / evaluation set data currently


    Methods
    -------
    pre_process_data()
        reads in the data and adds them into the `data_dict`. Used during initialisation.

    add_labels()
        reads in the labels data and adds them to the corresponding site (transcript_id, position).
        creates the gene_dict that maps the different sites to its corresponding gene_id. 
        Used during initialisation and if the self.train=True

    encode_kmer_sequence()
        computes all the possible 5mers and encodes all 66 of them. Used during initialisation.

    train_test_split()
        splits the data into training and testing data based on their gene_id. training size is from self.train_size.
        Used during initialisation and if self.train=True
    
    train_mode()
        sets self.eval=False. This allows the __getitem__ special to get sites from the train_transcript_position.
        Should only be used if self.train=True.

    eval_mode()
        sets self.eval=True. This allows the __getitem__ special to get sites from the test_transcript_position.
        Should only be used if self.train=True.

    __getitem__(index)
        special method that is required for Dataset subclasses. Gets the next item that is in our Dataset.

    __len__()
        special method that is required for Dataset subclasses. Returns the length of the dataset. 

    data_loader()
        returns a DataLoader object with the current Dataset.
    """

    def __init__(self, data_path, label_path=None, read_size=20, batch_size=128,
                 train=True, sgNex=False, train_size=0.8, seed=1800) -> None:
        """Constructor for the RNAData class

        Args:
            data_path (str): Path to the .json.gz file containing the data. Or .json file if we are looking at sgNex data.
            label_path (str, optional): Path to the .info file containing the labels, this will only be used if train=True. Defaults to None.
            read_size (int, optional): Number of reads that will be used per site. Defaults to 20.
            batch_size (int, optional): Number of sites we will use per batch for our Neural Network. Defaults to 128.
            train (bool, optional): Indicator if this Dataset should be a training set or not. Defaults to True.
            sgNext(bool, optional): Indicator if the  
            train_size (float, optional): Training size for the datset, this will only be used if train=True. Defaults to 0.8.
            seed (int, optional): Seed for reproducibility of results. Defaults to 1800.
        """
        self.data_path = data_path
        self.train = train
        self.read_size = read_size
        self.batch_size = batch_size
        self.sgNex = sgNex
        self.data_dict = {}
        self.fivemer_mapping = {}
        self.kmer_sequence_mapping = {}
        self.seed = seed

        print("Currently pre processing ...")
        self.pre_process_data()
        self.length = len(self.data_dict)
        print("Pre processing done!")

        if self.train:
            self.train_size = train_size
            self.gene_dict = {}
            self.train_transcript_position = []
            self.test_transcript_position = []
            self.eval = False # initialise as False

            print("Currently labelling ...")
            self.label_path = label_path
            self.add_labels()
            print('Labelling done')

            print("Currently Splitting ...")
            self.train_test_split()
            print('Splitting done')
            self.length = len(self.train_transcript_position)
        else:
            self.train_transcript_position = list(self.data_dict.keys())

        print("Currently encoding ...")
        self.encode_kmer_sequence()
        print('Encoding done')

    def pre_process_data(self):
        """
        Process the data that is in `self.data_path` and stores the relevant information in `self.data_dict`

        This is used in the constructor for initialisation.
        """

        # check if the path exists
        if not os.path.exists(self.data_path):
            raise Exception("Data path does not exist! Please check if the right data path is passed in")
        
        # check if the data path has the right extension
        if self.sgNex and not os.path.splitext(self.data_path)[1] == ".json":
            raise Exception("SgNex Data needs to end with .json extension, please ensure the right format is given")
        
        if not self.sgNex:
            _, gz_ext = os.path.splitext(self.data_path)
            filename, json_ext = os.path.splitext(_)

            if (json_ext+gz_ext) != ".json.gz":
                raise Exception("Data Paths needs to end with the extension .json.gz, please ensure the right format is given")


        # stores the intermediate data_dict values
        data_dict = {} 

        # processing for non sgNex data, need to unzip
        if not self.sgNex:
            with gzip.open(self.data_path, 'r') as f:
                for transcript in f:
                    data = json.loads(transcript)

                    for transcript_id, position_kmer in data.items():
                        for position, kmer_values in position_kmer.items():
                            position = int(position)
                            for kmer_sequence, values in kmer_values.items():

                                # initialise an empty dict for the transcript_id, position first
                                data_dict[(transcript_id, position)] = {}
                                data_dict[(transcript_id, position)]['kmer_sequence'] = kmer_sequence
                                data_dict[(transcript_id, position)]['signal_features'] = values
        # processing for sgNex, just read as json
        else:
            with open(self.data_path, 'r') as f:
                for transcript in f:
                    data = json.loads(transcript)


                    for transcript_id, position_kmer in data.items():
                        for position, kmer_values in position_kmer.items():
                            position = int(position)
                            for kmer_sequence, values in kmer_values.items():

                                # initialise an empty dict for the transcript_id, position first
                                data_dict[(transcript_id, position)] = {}
                                data_dict[(transcript_id, position)]['kmer_sequence'] = kmer_sequence
                                data_dict[(transcript_id, position)]['signal_features'] = values


        self.data_dict = data_dict

    def add_labels(self):
        """
        Process the data that is in `self.label_path`. Adds the label to the site in `self.data_dict` and also provides
        the pairings under `self.gene_dict`.

        This is used in the constructor for initialisation. This will only be called if `self.train=True`
        """

        # check if the path exists
        if not os.path.exists(self.label_path):
            raise Exception("Label path does not exist! Please check if the right label path is passed in")

        # iterate through the labels and record in the data dict
        data_info = pd.read_csv(self.label_path)
        for index, row in data_info.iterrows():
            transcript_id = row['transcript_id']
            position = int(row['transcript_position'])
            label = row['label']
            gene_id = row['gene_id']

            self.data_dict[(transcript_id, position)]['label'] = label 

            # creating mapping for the gene_dictionary
            if gene_id not in self.gene_dict:
                self.gene_dict[gene_id] = []
            self.gene_dict[gene_id].append((transcript_id, position))

    
    def encode_kmer_sequence(self):
        """
        Creates the encoding for each of the possible 5-mer and the corresponding 3 5-mer indices for each 7-mer.

        This is used in the constructor for initialisation. This will only be called if `self.train`=True
        """
        FLANKING_TERMS = ["A", "C", "G", "T"]
        POSSIBLE_CENTER_MOTIFS = [['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T']]
        # get all possible center motifs
        CENTER_MOTIF_SEQUENCES  = ["".join(x) for x in product(*POSSIBLE_CENTER_MOTIFS)]
        POSSIBLE_KMER_SEQUENCES = ["".join(x) for x in product(FLANKING_TERMS, CENTER_MOTIF_SEQUENCES, FLANKING_TERMS)]
        POSSIBLE_5MER_SEQUENCES = np.unique(np.array([[seq[i:i+5] for i in range(len(seq)-4)] for seq in POSSIBLE_KMER_SEQUENCES]).flatten())

        # helps to get the mappings for the 5mers 
        fivemer_mapping = {POSSIBLE_5MER_SEQUENCES[i] : i for i in range(len(POSSIBLE_5MER_SEQUENCES))}
        self.fivemer_mapping = fivemer_mapping

        # get the 3-mers for each 5-mer that we can have
        self.kmer_sequence_mapping = {seq : [fivemer_mapping[seq[i:i+5]] for i in range(len(seq)-4)] for seq in POSSIBLE_KMER_SEQUENCES}
    
    def train_test_split(self):
        """
        Splits the data into training set and testing set based on their `gene_id` with the training size being `self.train_size`

        This is used in the constructor for initialisation.

        Args:
            seed (int, optional): Random seed value for the splitting of the data. Defaults to 4266.
        """
        # computing the threshold amount
        threshold_amount = self.train_size * self.length

        # shuffling the gene_ids that are present in the dataset
        random.seed(self.seed)
        genes = list(self.gene_dict.keys())
        random.shuffle(genes)

        current_count = 0
        train_transcript_position = [] 

        # keep trying to add new genes to the training set until it hits the limit 
        while current_count < threshold_amount:
            current_gene = genes.pop()
            transcripts = self.gene_dict[current_gene]
            current_count += len(transcripts)
            train_transcript_position.extend(transcripts)


        test_transcript_position = [] 
        # store the remaining genes as the test genes
        for current_gene in genes:
            transcripts = self.gene_dict[current_gene]
            test_transcript_position.extend(transcripts)

        self.train_transcript_position = train_transcript_position
        self.test_transcript_position = test_transcript_position

        # empty it to reduce memory usage
        self.gene_dict = {}

    def train_mode(self):
        """
        Setter for `self.eval`=False so that we will make use of the training set data.

        This should only be used if `self.train`=True
        """
        # check that we have train and eval split first
        if self.train:
            self.eval = False
            self.length = len(self.train_transcript_position)

    def eval_mode(self):
        """
        Setter for `self.eval`=True so that we will make use of the evaluation set data.

        This should only be used if `self.train`=True
        """
        # check that we have train and eval split first
        if self.train:
            self.eval = True
            self.length = len(self.test_transcript_position)

    def __getitem__(self, index):
        if not self.train:
            np.random.seed(self.seed)
        if self.train and self.eval:
            # only if we are using eval (from training), we will use test_transcript
            id_position = self.test_transcript_position[index]
        else:
            # if we are using training data or just running predictions, we will use
            # train_transcript_position
            id_position = self.train_transcript_position[index]
        
        # get the relevant data for the current site
        site_data = self.data_dict[id_position]

        signal_features = site_data['signal_features']
        kmer_sequence = site_data['kmer_sequence']
        # extract the 3 5-mer indices for the current kmer
        fivemer_indices = np.tile(self.kmer_sequence_mapping[kmer_sequence], (self.read_size, 1))

        # select 20 samples without replacement
        if len(signal_features) < self.read_size:
            # if there isnt enough reads then we will sample with replacement
            indices = np.concatenate([np.arange(len(signal_features)), np.random.choice(len(signal_features), self.read_size - len(signal_features), replace=True)])
            selected_signals = np.array(signal_features)[indices, :]
        else:
            selected_signals = np.array(signal_features)[np.random.choice(len(signal_features), self.read_size, replace=False), :]

        x = np.concatenate((selected_signals, fivemer_indices), axis=1)
        x = torch.from_numpy(x)

        # only if we are training/evaluating, we will have labels
        if self.train:
            y = torch.from_numpy(np.array(site_data['label']))
            return x, y
        else:
            # we will return the site info as well when we are doing predictions 
            site = torch.from_numpy(np.array(index))
            return x, site


    def __len__(self):
        return self.length

    def data_loader(self):
        """
        Creates a `DataLoader` object for the current Dataset.
        
        There will be a weighted sampler if `self.train`=True and `self.eval`=False to ensure that we have a
        balanced class dataset when we are training.

        Returns:
            DataLoader: `DataLoader` object that helps to generate the data in the current Dataset which
            can be used by the Neural Network.
        """
        if not self.train:
            torch.manual_seed(self.seed)

        # if this is a training set of data, we will use a weighted sampler
        if self.train and not self.eval:
            # counting the number of positive and negative labels to know the weights to give them
            overall_labels = [self.data_dict[train_transcript]['label'] for train_transcript in self.train_transcript_position]
            positive_label = sum(overall_labels)
            negative_label = len(overall_labels) - positive_label
            class_sample_count = np.array([negative_label, positive_label])

            # weights are the inverse of the class sample count
            weights = 1.0 / class_sample_count
            sample_weights = weights[overall_labels]
            sample_weights = torch.from_numpy(sample_weights)
            
            sampler = WeightedRandomSampler(sample_weights,
                                            len(overall_labels),
                                            replacement=True)

            return DataLoader(self, batch_size=self.batch_size, sampler=sampler)

        # if we are evaluating / predicting then there is no need to oversample
        return DataLoader(self, batch_size=self.batch_size)