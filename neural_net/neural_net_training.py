# general purpose libraries
import numpy as np
import pandas as pd
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

class EarlyStopper:
    """
    Class that helps to determine if early stop is required for Neural Network training

    ...

    Attributes
    ----------
    patience : int
        Number of times we allow the `validation_roc_pr` to drop below (max_roc_pr - min_delta)
    min_delta : float
        Amount of buffer we allow the `validation_roc_rc` to drop below max_roc_pr and not consider it to be deviating
    counter : int
        Counter variable that helps to keep track of how many times the model has dropped below (max_roc_pr - min_delta) within the
        current max_roc_pr
    max_roc_pr : float
        Current best Max area under ROC and PRC (The score is computed based on the summation of Area under ROC and PRC)

    Methods
    -------
    early_stop(validation_roc_pr)
        Checks if the neural network should stop training. It should stop training if the `validation_roc_pr` has been lower than
        (max_roc_pr - min_delta) for more than the patience

    """

    def __init__(self, patience=3, min_delta=0):
        """Initialise the EarlyStopper object

        Args:
            patience (int, optional): Number of times we allow the `validation_roc_pr` to drop below (max_roc_pr - min_delta).
            Defaults to 3.

            min_delta (int, optional): Amount of buffer we allow the `validation_roc_rc` to drop below max_roc_pr and not consider it to be deviating.
            Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_roc_pr = float("-inf")
    
    def early_stop(self, validation_roc_pr):
        """
        Checks if the neural network should stop training. It should stop training if the `validation_roc_pr` has been lower than
        (max_roc_pr - min_delta) for more than the patience

        Args:
            validation_roc_pr (float): Summation of the current validation area under ROC and PRC

        Returns:
            bool: Whether the Neural Network should stop early
        """
        stop = False
        if validation_roc_pr > self.max_roc_pr:
            self.max_roc_pr = validation_roc_pr
            self.counter = 0
        elif validation_roc_pr < (self.max_roc_pr - self.min_delta):
            self.counter += 1
            stop = self.counter >= self.patience
        return stop

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
    optional.add_argument('-cpd', '--checkpoint-dict',
                        metavar='', type=str, default="",
                        help="Full filepath to the checkpoint dictionary, this is required if you want to continue training from the previous round (Default: '')")
    optional.add_argument('-erp', '--evalresults-path',
                        metavar='', type=str, default=os.path.join(".", "eval_result", "eval_res.csv"),
                        help="Full filepath to where the evaluation results should be saved to (Default: './eval_results/eval_res.csv')")
    optional.add_argument('-lr', '--learning-rate',
                        metavar='', type=float, default=0.01, help="Learning rate for the Neural Network (Default:0.01)")
    optional.add_argument('-wd', '--weight-decay',
                        metavar='', type=float, default=0.01, help="Weight Decay for the AdamW Optimiser (Default:0.01)")
    optional.add_argument('-bts', '--batch-size',
                        metavar='', type=int, default=128, help="Batch size for the Neural Network (Default: 128)")
    optional.add_argument('-rs', '--read-size',
                        metavar='', type=int, default=20,
                        help="Number of reads that is used per site for prediction by the Neural Network (Default: 20)")
    optional.add_argument('-ep', '--num-epochs',
                        metavar='', type=int, default=5, help="Number of epochs for training the Neural Network (Default: 5)")
    optional.add_argument('-ts', '--train-size',
                        metavar='', type=float, default=0.8, help="Size of training set, number between 0 and 1 (Default: 0.8)")
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
        for features, labels in tqdm(eval_rna_dataloader, total=len(eval_rna_dataloader)):

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
                       batch_size=args.batch_size, read_size=args.read_size,
                       train_size=args.train_size)
    rna_dataloader = rna_data.data_loader()

    # check if the user passed in the checkpoint_dict it should exist else raise Exception
    if args.checkpoint_dict and not (os.path.exists(args.checkpoint_dict)):
        raise Exception("Checkpoint directory stated does not exist, please check if the right directory is given")
    
    # assign the checkpoint value
    checkpoint = torch.load(args.checkpoint_dict) if args.checkpoint_dict else {}

    # creating the model object
    model = NeuralNetModel(batch_size=args.batch_size, read_size=args.read_size)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    # binary cross entropy loss function
    criterion = nn.BCELoss()
    # adamW optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']

    # check if the state dictionary exists, if it doesnt, create it
    if not (os.path.exists(os.path.dirname(args.modelstate_dict))):
        os.mkdir(os.path.dirname(args.modelstate_dict))

    # check if the eval results dictionary exists, if it doesnt, create it
    if not (os.path.exists(os.path.dirname(args.evalresults_path))):
        os.mkdir(os.path.dirname(args.evalresults_path))

    # creating the early stopper object
    early_stopper = EarlyStopper(patience=3, min_delta=0.05)
    # records the pr and roc score for each iteration
    pr = []
    roc = []

    print("Starting Training...")
    for epoch in range(args.num_epochs):
        # set to training mode
        model.train()
        rna_data.train_mode()
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

        # evaluate the curr model based on evaluation set
        # store the roc and pr results in the list that we are using to track and save the results
        roc_score, pr_score = evaluate_neural_net(model, rna_data)
        pr.append(pr_score)
        roc.append(roc_score)
        pd.DataFrame({'roc_score':roc, 'pr_score':pr}).to_csv(args.evalresults_path)
        
        # saving the model parameters
        if (roc_score + pr_score) >= early_stopper.max_roc_pr:
            # torch.save(model.state_dict(), args.modelstate_dict)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, args.modelstate_dict)
        # check if early stop is required
        if early_stopper.early_stop(roc_score + pr_score):
            break

    # evaluation
    print("Training Ended...")
    evaluate_neural_net(model, rna_data)


if __name__ == "__main__":
    torch.manual_seed(4266)
    args = parse_arguments()
    train_neural_net(args)