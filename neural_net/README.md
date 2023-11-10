# Team Ribosomes Neural Network Model

The `NeuralNetModel` is a `pytorch` neural network model that our team has developed for detection of m6a RNA modifications.

## Folder Structure

```
neural_net/
├── state/
│   └── sample_model.pth
├── __init__.py
├── neural_net_model.py
├── neural_net_pre_process.py
├── neural_net_pred.py
└── neural_net_training.py
```

* `state/` - Contains all the model states for the trained neural networks. It contains `sample_model.pth` which is a sample model state that has been pretrained based on the default settings of the `NeuralNetModel` and `neural_net_training.py`. 

* `neural_net_model.py` - Contains the `NeuralNetModel` neural network definition of the model

* `neural_net_pre_process.py` - Contains the `RNADataset` object that is a `Dataset` object and allows for pre processing of the RNA data used for training and prediction of the Neural Network.

* `neural_net_pred.py` - Contains the code for carrying out predictions on the trained `NeuralNetModel`

* `neural_net_training.py` - Contains the code for carrying out training on the `NeuralNetModel`

## Training the Model

Before training the model, ensure that you are in the current directory, `dsa4266_ribosomes/neural_net`. You can run the following command if you are in the `dsa4266_ribosomes` folder.

```bash
cd neural_net
```

To train the model, you can pass in the relevant arguments into the `neural_net_training` module and it will run the training process of the model.

**File**: `neural_net_training.py`

You can run the following command to find out the parameters that can be passed into the training the neural network.

*(If you are running on Windows, change `python3` to `python`)*

```python
python3 -m neural_net_training --help
```
## Flags that are available:

**Required Arguments:** These arguments needs to be passed in for the program to run

| short flag | long flag | description | type | 
|   :---:    |   :---:   | :---        | :--: | 
| `-dp` | `--data-path` | Full path to the `.json.gz` file containing the data for training | `str` |
| `-lp` | `--label-path` | Full path to the `.info` file containing the labels of the data used for training | `str` |

**Optional Arguments:** These arguments do not need to be passed in for the program to run

| short flag | long flag | description | type | defaults |
|   :---:    |   :---:   | :---        | :--: |   :--:   |
| `-msd` | `--modelstate-dict` | Full filepath to where we want to store the model state | `str` | **./state/model.pth** |
| `-cpd` | `--checkpoint-dict` | Full filepath to the checkpoint dictionary, this is required if you want to continue training from the previous round | `str` | **''** | 
| `-erp` | `--evalresults-path` | Full filepath to where the evaluation results should be saved to | `str` | **./eval_results/eval_res.csv** |
| `-lr` | `--learning-rate` | Learning rate for the Neural Network | `float` | **0.01** |
| `-wd` | `--weight-decay` | Weight Decay for the AdamW Optimiser | `float` | **0.01** |
| `-bts` | `--batch-size` | Batch size for the Neural Network | `int` | **128** | 
| `-rs` | `--read-size` | Read size for the Neural Network | `int` | **20** | 
| `-ep` | `--num-epochs` | Number of epochs for training the Neural Network | `int` | **5** | 
| `-ts` | `--train-size` | Size of training set, number between 0 and 1 | `float` | **0.8** | 

### Run training on your own dataset

You can make use of the following command in bash to run the training process for your model and pass in the relevant optional flags should you want to change any of the training parameters. Note that you will have to pass in `--data-path` and `--label-path` since they are required arguments.

*(If you are running on Windows, change `python3` to `python`)*

```bash
python3 -m neural_net_training --data-path <path_to_data_file> --label-path <path_to_label_data>
```
During training, the evaluation results (based on the Area under ROC and PRC) on the evaluation set will be saved in `./eval_result/eval_res.csv`. Note that this can be changed with the `--evalresults-path` flag. 

The `modelstate-dict` will also be saved as `./state/model.pth` which you can use in the prediction step below.

### Run training with sample dataset

To run the training model with the sample dataset: `../data/sample.json.gz`, you can run the following command in bash. You can pass in other optional flags as well should you want to change any of the training parameters.

*(If you are running on Windows, change `python3` to `python`)*

```bash
python3 -m neural_net_training --data-path "../data/sample.json.gz" --label-path "../data/sample.info"
```

During training, the evaluation results (based on the Area under ROC and PRC) on the evaluation set will be saved in `./eval_result/eval_res.csv`. Note that this can be changed with the `--evalresults-path` flag. 

The `modelstate-dict` will also be saved as `./state/model.pth` which you can use in the prediction step below.

## Making Predictions with the Model

Before training the model, ensure that you are in the current directory, `dsa4266/neural_net`. You can run the following command if you are in the `dsa4266` folder.

```bash
cd neural_net
```

**Note**: You can still make predictions without going through the above step of training the model. You can make use of the sample `modelstate-dict` that is stored within `./state/sample_model.pth`.

**File**: `neural_net_pred.py`

You can run the following command in bash to find out the parameters that can be passed into the prediction of the neural network.

*(If you are running on Windows, change `python3` to `python`)*

```bash
python3 -m neural_net_pred --help
```
### Flags that are available:

**Required Arguments:** These arguments needs to be passed in for the program to run

| short flag | long flag | description | type | 
|   :---:    |   :---:   | :---        | :--: | 
| `-msd` | `--modelstate-dict` | Full filepath to where the model state dict is stored during the Neural Network training | `str` |
| `-dp`  | `--data-path` | Full filepath to where the `.json.gz` file of the data that we want to predict is located | `str` |

**Optional Arguments:** These arguments do not need to be passed in for the program to run

| short flag | long flag | description | type | defaults |
|   :---:    |   :---:   | :---        | :--: |   :--:   |
| `-rf` | `--results-folder` | Full path to the directory where the result of the prediction should be stored. | `str` | **./results/** |
| `-bts`| `--batch-size` | Batch size that was used for training the Neural Network | `int` | **128** |
| `-rs` | `--read-size` | Read size that was used for training the Neural Network | `int` | **20** |
| `-snx`| `--sg-Nex` | Flag for if we are predicting on SgNex data | `bool` |**False** |

### Run predictions with your own dataset

You can make use of the following command in bash to run predictions on your model. Note that you will have to pass in `--modelstate-dict` and `--data-path` since they are required arguments.

*(If you are running on Windows, change `python3` to `python`)*

**Note**: You can make use of the sample `modelstate-dict` if you did not train your own model. The path to the sample model state is `./state/sample_model.pth`

```bash
python3 -m neural_net_pred --modelstate-dict <path_to_model_state_dict> --data-path <path_to_prediction_data>
```
After the prediction is ran, the prediction results will be in `./results/<data_path_prefix>.csv` where the prefix is the file name before `.json.gz`.

### Run predictions with sample dataset 

You can make use of the following command in bash to run predictions on the sample dataset.

*(If you are running on Windows, change `python3` to `python`)*

```bash
python3 -m neural_net_pred --modelstate-dict "./state/sample_model.pth"  --data-path "../data/sample.json.gz"
```
After the prediction is ran, the prediction results will be in `./results/sample.csv`.