<!-- omit in toc -->
# DSA4266 Project 2: Identification of RNA modifications from direct RNA-Seq data

This repository contains the codes that are created by Team Ribosome for the purposes of detecting m6A RNA modifications from direct RNA-Seq data.

<!-- omit in toc -->
## Table of Content:

- [Software Requirements](#software-requirements)
- [Setting Up](#setting-up)
  - [Machine Setup](#machine-setup)
  - [Cloning Repository](#cloning-repository)
  - [Install Software / Packages](#install-software--packages)
- [Running Model](#running-model)
- [Team Members](#team-members)

## Software Requirements

1. Python version $\geq 3.8$

## Setting Up

Do follow through the following steps carefully to ensure that the model can run smoothly. 

### Machine Setup 

1. **AWS Instance**: If you are running on the AWS Instance, start a *new* instance that is at least *t3.xlarge*. This is to ensure that the model does not run too long. Note that the default Python version for the instance will be at least **Python 3.8**.

2. **Locally** : If you are running locally, ensure that you have at least **Python** version $\geq 3.8$ 


### Cloning Repository

To clone the repository into your device, you can run the following command:

```bash
git clone https://github.com/xuan-jun/dsa4266_ribosomes.git
```

After it has been successfully cloned, you should see a folder `dsa4266_ribosomes` created.


### Install Software / Packages

After the repository has been cloned over successfully, run the following commands to install the relevant software or packages.

1. **Installing `pip`**

Note that when you are prompted to confirm, just enter "*y*"

```bash
sudo apt install python3-pip
```

2. **Installing packages required**

Ensure that you are within the `dsa4266_ribosomes` folder before running the following command

```bash
pip3 install -r requirements.txt
```

Once the packages are installed, you are ready to start running the training/predictions for the model!

## Running Model

Do ensure that you have completed all the steps above before trying to run the model.

The instructions for running the model is located <a href="./neural_net/" target="_blank">here</a> 

## Team Members

1. Chong Wan Fei  (**Handle**: [wanfeijong](https://github.com/wanfeijong))

2. Jonas Lim Wei Qi (**Handle**: [jonaslwq](https://github.com/jonaslwq))

3. Liew Siew Kit (**Handle**: [siewkit](https://github.com/siewkit))

4. Ng Xuan Jun (**Handle**: [xuan-jun](https://github.com/xuan-jun))