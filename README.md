# SancScreen
This code is accompanying our submission to LWDA22

Below, you will find instructions for installing the dependencies, as well as a description of each important source file to recreate our experiments.
In case of questions and / or problems with the code, feel free to open an issue on this repository.

# Overview
- The dataset files can be found in `/data/` and are structured as follows:
    - `sancscreen_c_{train,test}_{x,y}.npy` contain the training and test splits for the data points for which only class labels are present. This will be used for training and evaluating the Random Forest and Neural Network. Notice that we use 20 percent of the training set for validation when training the Neural Network.
    - `sancscreen_e_{x,y}.npy` contain the expert annotation
    - `feature_names.txt` contains all feature names in the same order as in the Numpy arrays
- You can find the code that trains the Neural Network in `code/train_model.py`, the model architecture in `code/model.py` and the training configuration in `code/parameters.py`
- `code/exp1.py` evaluates the trained models on the test dataset
- `code/exp2.py` compares different explainability methods on both models and plots the results (Figure 5 in the paper)
- `code/exp3.py` outlines the debugging use case and plots the results (Figure 7 in the paper)
- `code/exp4.py` calculates the significance values for `code/exp2.py` using a Wilcoxon test
- `code/exp5.py` calculates the correlations between the explanations and creates the plot in Figure 6.
- `code/dataset_viz.py` contains the code to generate the plots analyzing the dataset and expert annotations.

# Requirements
To recreate the experiments, first, install the dependencies using `python` version 3.8 or newer:

`pip install matplotlib==3.4.2 pandas==1.2.4 tqdm seaborn captum scipy shap==0.39.0 torch==1.8.1`

`pip install git+https://github.com/MatthiasJakobs/seedpy.git@932c7cac7e32bcaeb81e72658466db63ae7583a5`

To speed up performance, temporary files (model weights, training logs etc.) are generated.
If you want to recreate the experiments in their entirety, make sure to remove `results/checkpoints/*`, `results/*` and `results/exp2/*` before running the code.
