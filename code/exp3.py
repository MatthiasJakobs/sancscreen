# Debugging Usecase

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_sancscreen
from parameters import sancscreen_config

from exp2 import run_neural_network, run_random_forest

def plot_zero_bars(all_attr, baseline, path, all_labels, names):

    #plt.rcParams['font.size'] = '11'
    #plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(2, 1, sharey=True, figsize=(4.5, 5))
    #fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 4))

    width = 0.8
    for idx in range(len(all_attr)):
        attr = all_attr[idx]

        sorted_inds = np.argsort(-np.array(attr))

        ax[idx].set_title(f"{names[idx]}")
        ax[idx].set_ylim(0, 1)
        ax[idx].set_ylabel("Explanations close to 0 (percent)")

        shadings = ["sandybrown", "violet", "salmon", "lightblue", "palegreen", "lightgrey"]
        edges = ["saddlebrown", "purple", "darkred", "blue", "green", "gray"]

        print(np.array(attr)[sorted_inds])
        sorted_labels = [all_labels[idx][k] for k in sorted_inds]
        sorted_edges = [edges[k] for k in sorted_inds]
        sorted_shadings = [shadings[k] for k in sorted_inds]

        ax[idx].bar(np.arange(len(attr)), np.array(attr)[sorted_inds], tick_label=sorted_labels, edgecolor=sorted_edges, color=sorted_shadings)
        ax[idx].set_xticklabels(sorted_labels, rotation = 12)

        
    plt.tight_layout()
    plt.savefig(path)

if __name__ == "__main__":
    # Load pretrained model, dataset and configs
    d = load_sancscreen()
    c = sancscreen_config
    rng = np.random.RandomState(123)
    feature_names = ["Indicator_value",	"Number_abnormalities",	"hidden_feature_3", "City_hit",	"Account_hit", "Name_hit", "Other_hit",	"hidden_feature_1",	"Country_hit",	"Address_hit",	"Sanction_hit",	"hidden_feature_2",	"Bank_hits", "AccountOwner_hit", "Text_hits", "Specific_Country_hit", "Unique_hits", "Different_hits", "Number_countries"]

    X_train = d.c_train_x.numpy()
    y_train = d.c_train_y.numpy()
    X_test = d.c_test_x.numpy()
    y_test = d.c_test_y.numpy()

    limit_to = -1
    X = d.e_x.numpy()
    y = d.e_y.numpy()
    gt_attributions = d.annot.numpy()

    perm = rng.permutation(np.arange(len(X)))[:limit_to]
    X = X[perm]
    y = y[perm]
    gt_attributions = gt_attributions[perm]

    rf_labels, rf_attributions = run_random_forest(X_train, y_train, X_test, y_test, X, y, gt_attributions, feature_names)
    nn_labels, nn_attributions = run_neural_network(X_train, y_train, X_test, y_test, X, y, gt_attributions, feature_names, c)

    expert_says_zero = np.mean(gt_attributions == 0, axis=0)
    print(expert_says_zero)

    # Candidates: #17 (96 percent) (index 16)
    print(expert_says_zero[16])

    print("RANDOM FOREST")
    rf_zeros = []
    for (method, attr) in zip(rf_labels, rf_attributions):
        #print(np.isclose(attr, np.zeros_like(attr)))
        method_says_zero = np.mean(np.isclose(attr, np.zeros_like(attr), atol=0.01, rtol=0), axis=0)[16]
        print(method, method_says_zero, attr.shape)
        rf_zeros.append(method_says_zero)

    print("NEURAL NETWORK")
    nn_zeros = []
    for (method, attr) in zip(nn_labels, nn_attributions):
        print(method, attr.shape)
        method_says_zero = np.mean(np.isclose(attr, np.zeros_like(attr), atol=0.01, rtol=0), axis=0)[16]
        print(method, method_says_zero, attr.shape)
        nn_zeros.append(method_says_zero)

    print(rf_labels)
    plot_zero_bars([rf_zeros, nn_zeros], 0.5, "plots/zero-bars.pdf", [rf_labels, nn_labels], ["Random Forest","Neural Network"])
