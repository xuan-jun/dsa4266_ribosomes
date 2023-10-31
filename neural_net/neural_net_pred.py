# general purpose libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# libraries for neural networks
import torch
import torch.nn as nn

# model libraries
from neural_net_model import NeuralNetModel
from neural_net_pre_process import RNAData

def parse_arguments():
    """Parses the arguments that are supplied by the user

    Returns:
        Namespace: parsed arguments from the user
    """
    parser = argparse.ArgumentParser(description="Runs the prediction for the trained Neural Network of m6a modifications")
    required = parser.add_argument_group("required arguments")
    required.add_argument("-msd", "--modelstate-dict", type=str, metavar="",
                        required=True, help="Full filepath to where the model state dict is stored during the Neural Network training")
    required.add_argument("-dp", "--data-path", type=str, metavar="", 
                        required=True, help="Full filepath to where the .json.gz file of the data that we want to predict is located")
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("-rf", "--results-folder", type=str, default=os.path.join(".", "results"), metavar="",
                        help="Full path to the directory where the result of the prediction should be stored. (Default: ./results/)")
    optional.add_argument("-bts", "--batch-size", type=int, default=128, metavar="",
                        help="Batch size that was used for training the Neural Network (Default: 128)")
    optional.add_argument("-rs", "--read-size", type=int, default=20 , metavar="", 
                        help="Read size that was used for training the Neural Network (Default: 20)")
    args = parser.parse_args()
    return args


def predict_neural_network(args):
    """Runs predictions on a trained NeuralNetModel based on the arguments that are passed in by the user

    Args:
        args (Namespace): contains the arguments that are passed in by the user through argparser
    """

    # check if the path model state exists
    if not (os.path.exists(args.modelstate_dict)):
        raise Exception("Model state does not exist, please check the path or if training has been done!")

    # initialising the model and dataloader
    rna_data = RNAData(data_path=args.data_path, read_size=args.read_size,
                       batch_size=args.batch_size, train=False)
    rna_dataloader = rna_data.data_loader()

    # initialising the neural net model
    model = NeuralNetModel(batch_size=args.batch_size, read_size=args.read_size)
    
    # loading the model_state_dict
    checkpoint = torch.load(args.modelstate_dict)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # set to evaluation mode

    # create the results directory if it doesnt exist
    if not (os.path.exists(args.results_folder)):
        os.mkdir(args.results_folder)

    # creating the results file name based on the original data value 
    results_file_name = f"{os.path.basename(args.data_path).split('.')[0]}.csv"

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

    # extracting the transcript_id and transcript_position of the predictions
    rna_test_sites = rna_data.train_transcript_position
    transcript_id = list(map(lambda x : rna_test_sites[x][0], site_indices))
    transcript_position = list(map(lambda x : rna_test_sites[x][1], site_indices))

    final_data = pd.DataFrame({"transcript_id" : transcript_id,
                            "transcript_position" : transcript_position,
                            "score" : pred_scores})
    final_data.to_csv(os.path.join(args.results_folder, results_file_name), index=False)
    print("Output file saved!")

if __name__ == "__main__":
    torch.manual_seed(4266)
    args = parse_arguments()
    predict_neural_network(args)