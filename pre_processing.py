import gzip
import pandas as pd
import numpy as np
import json
import glob
import os

def pre_processing():
    
    ## TODO:
    ### parse the json data
    ### aggregate across each position (since each position has multiple reads)
    ### return an aggregated data
    ### split into training and testing data based on geneID

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
                            curr_row = [transcript_id, curr_position, curr_kmer, num_of_observations]
                            curr_row.extend(curr_aggregated_values)

                            processed_data.append(curr_row)

    processed_data = pd.DataFrame(processed_data, columns = ["transcript_id", "position", "kmer_sequence", "num_observations",
                                                    "dwelling_length", "sd_signal", "mean_signal"])

    return processed_data