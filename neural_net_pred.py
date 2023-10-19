# libraries for processing
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# libraries for neural networks
import torch
import torch.nn as nn

# importing Neural Network Model
from neural_net_model import NeuralNetModel
from neural_net_pre_process import RNAData

# CONSTANTS
MODEL_STATE_DICT = os.path.join(".", "state", "model.pth") 
DATA_FOLDER_PATH = os.path.join(".", "data") 
RESULTS_FOLDER_PATH = os.path.join(".", "results")
DATA_FILE_NAME = "dataset0.json.gz"
BATCH_SIZE = 128 # note that this needs to be the same the training process
READ_SIZE = 20 # note that this needs to be the same the training process

if not (os.path.exists(MODEL_STATE_DICT)):
    raise Exception("Model state does not exist, please check the path or if training has been done!")

# initialising the model and dataloader
rna_data = RNAData(data_path=os.path.join(DATA_FOLDER_PATH, DATA_FILE_NAME),
                   read_size=READ_SIZE, batch_size=BATCH_SIZE, train=False)
rna_dataloader = rna_data.data_loader()

model = NeuralNetModel(batch_size=BATCH_SIZE, read_size=READ_SIZE)
model.load_state_dict(torch.load(MODEL_STATE_DICT))
model.eval() # set to evaluation mode

# create the results directory
if not (os.path.exists(RESULTS_FOLDER_PATH)):
    os.mkdir(RESULTS_FOLDER_PATH)

# creating the results file name based on the original data value 
results_file_name = f"{DATA_FILE_NAME.split('.')[0]}.csv"

## running the test 

# stores the predicted_scores for each evaluation data
pred_scores = torch.tensor([])
site_indices = []

# run without computing gradients
print("Predicting now...")
with torch.no_grad():
    
    # loop through testing data
    for features, index in tqdm(rna_dataloader, total=len(rna_dataloader)):

        index = index.flatten().numpy()
        # keep track of the indices
        site_indices.extend(index)

        # use the model to predict the values
        y_pred = model(features)

        # combining the previous scores and the current scores
        pred_scores = torch.concat([pred_scores, y_pred])
    
print("Preparing output file...")
# tidying up the data and converting into csv
pred_scores = pred_scores.numpy().flatten().astype(float)

rna_test_sites = rna_data.train_transcript_position
transcript_id = list(map(lambda x : rna_test_sites[x][0], site_indices))
transcript_position = list(map(lambda x : rna_test_sites[x][1], site_indices))

final_data = pd.DataFrame({"transcript_id" : transcript_id,
                           "transcript_position" : transcript_position,
                           "score" : pred_scores})
final_data.to_csv(os.path.join(RESULTS_FOLDER_PATH, results_file_name), index=False)
