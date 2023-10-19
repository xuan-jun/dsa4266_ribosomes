# general purpose libraries
import numpy as np
import os
from tqdm import tqdm

# torch libraries
import torch
import torch.nn as nn
import torchmetrics
from torcheval.metrics.functional import binary_auprc

# model libraries
from neural_net_model import NeuralNetModel
from neural_net_pre_process import RNAData

# CONSTANTS
DATA_PATH = os.path.join(".", "data", "dataset0.json.gz")
LABEL_PATH = os.path.join(".", "data", "data.info")
MODEL_STATE_DIR = os.path.join(".", "state")
MODEL_STATE_FILENAME = "model_param.pth"
LEARNING_RATE = 0.001
BATCH_SIZE = 128
READ_SIZE = 20
NUM_EPOCHS = 11 

# preparing the data
rna_data = RNAData(data_path=DATA_PATH, label_path=LABEL_PATH, batch_size=BATCH_SIZE, read_size=READ_SIZE)
print(f"Length of training data: {len(rna_data)}")
rna_dataloader = rna_data.data_loader()

# initalizing the model
model = NeuralNetModel(batch_size=BATCH_SIZE, read_size=READ_SIZE)

# binary cross entropy loss function
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# check if the state dictionary exists, if it doesnt, create it
if not (os.path.exists(MODEL_STATE_DIR)):
    os.mkdir(MODEL_STATE_DIR)

print("Starting Training...")
for epoch in range(NUM_EPOCHS):
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

        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())

        # saving the model parameters
        torch.save(model.state_dict(), os.path.join(MODEL_STATE_DIR, MODEL_STATE_FILENAME))

# evaluation

model = NeuralNetModel(batch_size=BATCH_SIZE, read_size=READ_SIZE)
model.load_state_dict(torch.load(os.path.join(MODEL_STATE_DIR, MODEL_STATE_FILENAME)))
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
print(f"AU ROC Score: {auroc(pred_scores, lab).item():.4f}")
print(f"AU PRC Score : {binary_auprc(pred_scores, lab).item():.4f}")
