# Team Ribosomes Neural Network Model

The `NeuralNetModel` is a `pytorch` neural network model.

## Folder Structure

```
neural_net/
├── __init__.py
├── neural_net_model.py
├── neural_net_pre_process.py
├── neural_net_pred.py
├── neural_net_training.py
└── requirements.txt
```

* `neural_net_model.py` - Contains the `NeuralNetModel` neural network definition of the model

* `neural_net_pre_process.py` - Contains the `RNADataset` object that is a `Dataset` object and allows for pre processing of the RNA data used for training and prediction of the Neural Network.

* `neural_net_pred.py` - Contains the code for carrying out predictions on the trained `NeuralNetModel`

* `neural_net_training.py` - Contains the code for carrying out training on the `NeuralNetModel`

* `requirements.txt` - Specifies the libraries required for the program to run

## Setup

1. Ensure that you have **Python** installed on your device

2. Install the necessary libraries required. You can run the following bash command

```bash
pip install -r requirements.txt
```

## Training the Model

To train the model, you can pass in the relevant arguments into the `neural_net_training` module and it will run the training process of the model.

**File**: `neural_net_training.py`

You can run the following command to find out the parameters that can be passed into the training the neural network.

```python
python -m neural_net_training --help
```
## Flags that are available:

**Required Arguments**

* `-dp` or `--data-path` - Full path to the `.json.gz` file containing the data for training

* `-lp` or `--label-path` - Full path to the `.info` file containing the labels of the data used for training

**Optional Arguments**

* `-msd` or `--modelstate-dict` - Full filepath to where we want to store the model state (*Default: ./state/model.pth*) 

* `-cpd` or `--checkpoint-dict` - Full filepath to the checkpoint dictionary, this is required if you want to continue training from the previous round (*Default: ''*) 

* `-erp` or `--evalresults-path` - Full filepath to where the evaluation results should be saved to (*Default: ./eval_results/eval_res.csv*) 

* `-lr` or `--learning-rate` - Learning rate for the Neural Network (*Default: 0.01*) 

* `-wd` or `--weight-decay` - Weight Decay for the AdamW Optimiser (*Default:0.01*) 

* `-bts` or `--batch-size` - Batch size for the Neural Network (*Default: 128*) 

* `-rs` or `--read-size` - Read size for the Neural Network (*Default: 20*) 

* `-ep` or `--num-epochs` - Number of epochs for training the Neural Network (*Default: 5*) 

* `-ts` or `--train-size` - Size of training set, number between 0 and 1 (*Default: 0.8*) 

### Example command

You can make use of the following command in bash to run the training process for your model. Note that you will have to pass in `--data-path` and `--label-path` since they are required arguments.

```bash
python -m neural_net_training --data-path <path_to_data_file> \
    --label-path <path_to_label_data>
```

## Making Predictions with the Model

Note that you can only make predictions after you have trained the model

**File**: `neural_net_pred.py`

You can run the following command in bash to find out the parameters that can be passed into the prediction of the neural network.

```bash
python -m neural_net_pred --help
```
### Flags that are available:

**Required Arguments**

* `-msd` or `--modelstate-dict` - Full filepath to where the model state dict is stored during the Neural Network training 

* `-dp` or `--data-path` - Full filepath to where the `.json.gz` file of the data that we want to predict is located 

**Optional Arguments**

* `-rf` or `--results-folder` - Full path to the directory where the result of the prediction should be stored. (*Default: ./results/*)

* `-bts` or `--batch-size` - Batch size that was used for training the Neural Network (*Default: 128*)

* `-rs` or `--read-size` - Read size that was used for training the Neural Network (*Default: 20*)

### Example command

You can make use of the following command in bash to run predictions on your model. Note that you will have to pass in `--modelstate-dict` and `--data-path` since they are required arguments.

```bash
python -m neural_net_pred --modelstate-dict <path_to_model_state_dict> \
    --data-path <path_to_prediction_data>
```