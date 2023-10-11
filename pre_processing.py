import gzip
import pandas as pd
import numpy as np
import json
import glob
import os
from sklearn.model_selection import GroupShuffleSplit 

def pre_processing_separated_kmer():
    """Processes the zipped .gz data and returns it in a tabular format. This function works on the 5 length sequence and places
    each of the flanking terms as a separate record themselves. 

    Returns:
        pd.DataFrame: Contains the following columns:
        
        [`transcript_id`, `transcript_position`, `reference_position`, `kmer_sequence`, `original_kmer_sequence`, `dwelling_length`, `sd_signal`, `mean_signal`].

        Note that each row is a single sample that is gathered from the raw data and we will .

        Note that the `kmer_sequence` is a 5 length sequence (depending on which part of the original kmer sequence it is part of)
    """

    file_path = glob.glob(os.path.join("data", "*.gz"))[0] # extracts the files with the gz extension and pick the first one
    processed_data = [] # keeps all our processed data

    # unzipping the gz file
    with gzip.open(file_path,'r') as f:
        # process each of the transcript

        for transcript in f:
            data = json.loads(transcript)
            
            # iterate over each of the transcript id and {position: {"kmer"} : [[]]}
            for transcript_id, position_kmer in data.items():
                # iterate over each of the position and {"kmer" : [[]]}
                for position, kmer_values in position_kmer.items():
                    # iterate over each of the kmer and the features
                    for kmer, values in kmer_values.items():
                        original_kmer = kmer[1:6] # extracts the original kmer (centre)

                        positions = [-1, 0, 1] # used for iterating over each of the positions
                        kmer_positions = [(0, 5), (1, 6), (2, 7)] # used for iterating the kmer_positions
                        values_to_take = [(0, 3), (3, 6), (6, 9)]
                        num_of_observations = len(values) # computes the number of observations for each of the transcript_id

                        # across [length_signal1, sd_signal1, mean_signal1, length_signal2, sd_signal2, mean_signal2, length_signal3, sd_signal3, mean_signal3]
                        # reshape the data so that it becomes [length, sd, mean_signal]
                        aggregated_values = np.array(values).reshape(-1, 3)

                        # contains the transcript_id, position, curr_kmer for the 3 possible 5mer
                        specific_kmer_info = []
                        # iterate through all 3 possible 5-mer
                        for i in range(3):
                            curr_position = int(position) + positions[i]
                            curr_kmer = kmer[kmer_positions[i][0]:kmer_positions[i][1]]
                            curr_row = [transcript_id, curr_position, int(position), curr_kmer, original_kmer]

                            specific_kmer_info.append(curr_row)

                        # repeats the sequence of 3 5-mer for all the observations
                        kmer_info_all = np.tile(np.array(specific_kmer_info), num_of_observations).reshape(-1, 5)

                        # column bind the kmer_info and the values for them
                        combined_data = np.column_stack((kmer_info_all, aggregated_values)).tolist() 
                        processed_data.extend(combined_data)
                        
    processed_data = pd.DataFrame(processed_data, columns = ["transcript_id", "transcript_position", "reference_position",
                                                             "kmer_sequence", "original_kmer_sequence",
                                                             "dwelling_length", "sd_signal", "mean_signal"])
    
    return processed_data

def pre_processing_entire_kmer():
    """Processes the zipped .gz data and returns it in a tabular format. This function works on the 7 length sequence and each sample
    will just be a record by itself.

    Returns:
        pd.DataFrame: Contains the following columns:

        [`transcript_id`, `transcript_position`, `kmer_sequence`, `dwelling_length1`, `sd_signal1`, `mean_signal1`, `dwelling_length2`, `sd_signal2`, `mean_signal2`, `dwelling_length3`, `sd_signal3`, `mean_signal3`].

        Note that each row is a single sample that is gathered from the raw data.

        Note that the kmer_sequence is a 7 length sequence
    """
    file_path = glob.glob(os.path.join("data", "*.gz"))[0] # extracts the files with the gz extension and pick the first one
    processed_data = [] # keeps all our processed data

    # unzipping the gz file
    with gzip.open(file_path,'r') as f:
        # process each of the transcript

        for transcript in f:
            
            data = json.loads(transcript)
            
            # iterate over each of the transcript id and {position: {"kmer"} : [[]]}
            for transcript_id, position_kmer in data.items():
                # iterate over each of the position and {"kmer" : [[]]}
                for position, kmer_values in position_kmer.items():
                    position = int(position)
                    # iterate over each of the kmer and the features
                    for kmer, values in kmer_values.items():
                        num_observations = len(values)

                        # reshape so that the columns are [length_signal1, sd_signal1, mean_signal1, 
                        # length_signal2, sd_signal2, mean_signal2, length_signal3, sd_signal3, mean_signal3] and each row is a sample
                        aggregated_values = np.array(values).reshape(-1, 9)

                        # repeat the kmer data for the number of observations
                        kmer_data = np.array([transcript_id, position, kmer])
                        repeated_data = np.tile(kmer_data, num_observations).reshape(-1, 3) 

                        # bind the data
                        combined_data = np.column_stack((repeated_data, aggregated_values)).tolist()

                        processed_data.extend(combined_data)
                        
    processed_data = pd.DataFrame(processed_data, columns = ["transcript_id", "transcript_position", "kmer_sequence", 
                                                             "dwelling_length1", "sd_signal1", "mean_signal1", 
                                                             "dwelling_length2", "sd_signal2", "mean_signal2",
                                                             "dwelling_length3", "sd_signal3", "mean_signal3"])
    
    processed_data['transcript_position'] = processed_data["transcript_position"].astype(int)
    
    return processed_data



def pre_processing_aggregated():
    """Processes the zipped .gz data and returns it in a tabular format. This function works on the 5 length sequence and places
    each of the flanking terms as a separate record themselves.
    
    However for this we will aggregate across all samples for a particular `transcript_id`. 

    Returns:
        pd.DataFrame: Contains the following columns: 

        [`transcript_id`, `position`, `reference_position`, `kmer_sequence`, `original_kmer_sequence`, `num_observations`, `dwelling_length`, `sd_signal`, `mean_signal`].

        Note that each row is an aggregation of the observations for that `transcript_id` and `position` gathered from the raw data.
    """

    file_path = glob.glob(os.path.join("data", "*.gz"))[0] # extracts the files with the gz extension and pick the first one
    processed_data = [] # keeps all our processed data

    # unzipping the gz file
    with gzip.open(file_path,'r') as f:
        # process each of the transcript

        for transcript in f:
            data = json.loads(transcript)
            
            # iterate over each of the transcript id and {position: {"kmer"} : [[]]}
            for transcript_id, position_kmer in data.items():
                # iterate over each of the position and {"kmer" : [[]]}
                for position, kmer_values in position_kmer.items():
                    # iterate over each of the kmer and the features
                    for kmer, values in kmer_values.items():
                        original_kmer = kmer[1:6] # extracts the original kmer (centre)

                        positions = [-1, 0, 1] # used for iterating over each of the positions
                        kmer_positions = [(0, 5), (1, 6), (2, 7)] # used for iterating the kmer_positions
                        values_to_take = [(0, 3), (3, 6), (6, 9)]
                        num_of_observations = len(values) # computes the number of observations for each of the transcript_id

                        # takes the mean of the values column wise and we will get an aggregated list of values
                        # across [length_signal1, sd_signal1, mean_signal1, length_signal2, sd_signal2, mean_signal2, length_signal3, sd_signal3, mean_signal3]
                        aggregated_values = list(pd.DataFrame(values).mean(axis = 0))

                        # iterate through all 3 possible 5-mer
                        for i in range(3):
                            curr_position = int(position) + positions[i]
                            curr_kmer = kmer[kmer_positions[i][0]:kmer_positions[i][1]]
                            curr_aggregated_values = aggregated_values[values_to_take[i][0]:values_to_take[i][1]]                      
                            curr_row = [transcript_id, curr_position, int(position), curr_kmer, original_kmer, num_of_observations]
                            curr_row.extend(curr_aggregated_values)

                            processed_data.append(curr_row)

    processed_data = pd.DataFrame(processed_data, columns = ["transcript_id", "position", "reference_position",
                                                             "kmer_sequence", "original_kmer_sequence", "num_observations",
                                                             "dwelling_length", "sd_signal", "mean_signal"])

    return processed_data

def add_labels(processed_data):
    """Takes in a `pd.DataFrame` of the pre processed data from either the following functions:
    `pre_processing_separated_kmer()`,`pre_processing_entire_kmer()`, `pre_processing_aggregated()`
    and merges it with the labels file. Adding the `gene_id` and `label` for whether there are mutations

    Args:
        processed_data (pd.DataFrame): Processed data from either `pre_processing_separated_kmer()`,`pre_processing_entire_kmer()`, `pre_processing_aggregated()`

    Returns:
        pd.DataFrame: Contains the additional columns of "gene_id" and "label" added to the processed_data
    """

    data_path = glob.glob(os.path.join("data", "*.info"))[0] # extracts the files with the info extension and pick the first one

    # read in the data file
    data_info = pd.read_csv(data_path)

    if "reference_position" in processed_data.columns:
        processed_data.rename(columns = {"reference_position" : "transcript_position"}, inplace=True)

    # merging the data
    final_data = processed_data.merge(data_info, how="inner", on=["transcript_id", "transcript_position"])

    return final_data


def add_features(labelled_data):
    """Adds in 2 additional features to the labelled data. 1) The count of each nucleotide, 2) The count of each dinucleotide

    Args:
        labelled_data (pd.DataFrame): labelled data from the `add_labels` function 

    Returns:
        pd.DataFrame: labelled_data but with an addition of 20 columns (4 from nucleotides and 16 from dinuclotides)
    """

    # creating all the possible nucleotides
    nucleotides = ["A", "C", "G", "T"]
    dinucletides = [i + j for i in nucleotides for j in nucleotides]

    # count the frequency of each nucleotide
    for n in nucleotides:
        labelled_data[f"{n}_count"] = labelled_data['kmer_sequence'].apply(lambda x : x.count(n))

    # count the frequney of each dinucleotide
    for dn in dinucletides:
        # gets the count of the dinucleotide for each of the consecutive 2 sequence of the kmer sequence
        labelled_data[f"{dn}_count"] = labelled_data['kmer_sequence'].apply(
            lambda seq : sum([seq[i:i+2].count(dn) for i in range(len(seq) - 1)]))

    return labelled_data



def train_test_split(labelled_data, random_state, test_size=0.2):
    """Splits the labelled data into training testing

    Args:
        labelled_data (pd.DataFrame): labelled data from `add_labels()` or the data could have added features from `add_features()`
        random_state (int): random integer value for the splitting state
        test_size (float, optional): Size of the test set. Defaults to 0.2.

    Returns:
        (`train_x`, `train_y`, `test_x`, `test_y`): Training data features, Training data labels, Testing data features, Testing data labels   
        
    """
    
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state = random_state)
    # split across the gene_id
    split = splitter.split(labelled_data, groups=labelled_data['gene_id'])
    train_inds, test_inds = next(split)

    train = labelled_data.iloc[train_inds]
    train_x = train[train.columns.difference(["label"])]
    train_y = train[["label"]]
    test = labelled_data.iloc[test_inds]
    test_x = test[test.columns.difference(["label"])]
    test_y = test[["label"]]

    return (train_x, train_y, test_x, test_y)