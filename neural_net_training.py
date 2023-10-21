# general purpose libraries
import numpy as np
import os
from tqdm import tqdm
import argparse

# torch libraries
import torch
import torch.nn as nn
import torchmetrics
from torcheval.metrics.functional import binary_auprc

# model libraries
from neural_net_model import NeuralNetModel
from neural_net_pre_process import RNAData

def parse_arguments():
    """Parses the arguments that are supplied by the user

    Returns:
        Namespace: parsed arguments from the user
    """
    parser = argparse.ArgumentParser(description="Trains the Neural Network for predicting m6a modifications")
    required = parser.add_argument_group("required arguments")
    required.add_argument('-dp', '--data-path',
                        metavar='', type=str, required=True,help="Full path to the .json.gz file containing the data for training")
    required.add_argument('-lp', '--label-path',
                        metavar='', type=str, required=True, help="Full path to the .info file containing the labels of the data used for training")
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument('-msd', '--modelstate-dict',
                        metavar='', type=str, default=os.path.join(".", "state", "model.pth"),
                        help="Full filepath to where we want to store the model state (Default: ./state/model.pth)")
    optional.add_argument('-lr', '--learning-rate',
                        metavar='', type=float, default=0.001, help="Learning rate for the Neural Network (Default:0.001)")
    optional.add_argument('-bts', '--batch-size',
                        metavar='', type=int, default=64, help="Batch size for the Neural Network (Default: 64)")
    optional.add_argument('-rs', '--read-size',
                        metavar='', type=int, default=20,
                        help="Number of reads that is used per site for prediction by the Neural Network (Default: 20)")
    optional.add_argument('-ep', '--num-epochs',
                        metavar='', type=int, default=30, help="Number of epochs for training the Neural Network (Default: 30)")
    args = parser.parse_args()
    return args

def evaluate_neural_net(model, rna_data):
    """Evaluates the performance of the Neural Network based on the Area under ROC and PRC

    Args:
        model (NeuralNetModel): neural network model that was used for training
        rna_data (RNAData): RNAData object that was created during the training process

    Returns:
        tuple: tuple pair with the area under ROC curve and area under PR curve
    """
    # turn the model to evaluation mode
    model.eval()

    # stores the predicted_scores and actual labels for each evaluation data
    pred_scores = torch.tensor([])
    lab = torch.tensor([]).type(torch.float32)

    print("Starting Evaluation...")
    # run without computing gradients
    with torch.no_grad():
        
        # get the evaluation data and data loader
        rna_data.eval_mode()
        eval_rna_dataloader = rna_data.data_loader()

        # loop through eval data
        for features, labels in eval_rna_dataloader:

            labels = labels.flatten().type(torch.float32)
            # keep track of all the labels
            lab = torch.cat([lab, labels])

            # use the model to predict the values
            y_pred = model(features)

            # combining the previous scores and the current scores
            pred_scores = torch.cat([pred_scores, y_pred])

    # evaluating the ROC and PR AUC Scores
    auroc = torchmetrics.AUROC(task="binary")
    roc_score = auroc(pred_scores, lab).item()
    pr_score = binary_auprc(pred_scores, lab).item()
    print(f"AU ROC Score: {roc_score:.4f}")
    print(f"AU PRC Score : {pr_score:.4f}")

    return (roc_score, pr_score)


def train_neural_net(args):
    """Trains a NeuralNetModel based on the arguments that are passed in by the user

    Args:
        args (Namespace): contains the arguments that are passed in by the user through argparser
    """

    # preparing the data
    rna_data = RNAData(data_path=args.data_path, label_path=args.label_path,
                       batch_size=args.batch_size, read_size=args.read_size)
    print(f"Length of training data: {len(rna_data)}")
    rna_dataloader = rna_data.data_loader()

    # initalizing the model
    model = NeuralNetModel(batch_size=args.batch_size, read_size=args.read_size)

    # binary cross entropy loss function
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # check if the state dictionary exists, if it doesnt, create it
    if not (os.path.exists(os.path.dirname(args.modelstate_dict))):
        os.mkdir(os.path.dirname(args.modelstate_dict))

    print("Starting Training...")
    for epoch in range(args.num_epochs):
        # set to training mode
        model.train()
        loop = tqdm(enumerate(rna_dataloader),
                    total=len(rna_dataloader))
        for i, (x, y) in loop:
            label = y.flatten().type(torch.float32)
            
            # forward pass
            outputs = model(x)
            loss = criterion(outputs, label)

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loop.set_description(f"Epoch [{epoch+1}/{args.num_epochs}]")
            loop.set_postfix(loss=loss.item())

            # saving the model parameters
            torch.save(model.state_dict(), args.modelstate_dict)

    # evaluation
    evaluate_neural_net(model, rna_data)


if __name__ == "__main__":
    args = parse_arguments()
    train_neural_net(args)