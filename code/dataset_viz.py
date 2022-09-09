import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from parameters import sancscreen_config
from datasets import load_sancscreen

c = sancscreen_config
d = load_sancscreen()

colors = ["sandybrown", "violet", "salmon", "lightblue", "palegreen", "lightgrey"]

def distribution_of_binary_features():
    distribution = np.zeros((len(d.binary_features), 2)).astype(np.int32)
    for i, index in enumerate(d.binary_features):
        feats = np.concatenate([d.c_train_x[:, index].squeeze().numpy(), d.c_test_x[:, index].squeeze().numpy(), d.e_x[:, index].squeeze().numpy()])
        distribution[i] = np.bincount(feats.astype(np.int8))

    # convert from absolute to percentages
    distribution = distribution / np.sum(distribution, axis=1).reshape(-1, 1)

    readable_feature_names = ["\n".join(name.split("_")) for name in np.array(d.feature_names)[d.binary_features]]

    fig, axs = plt.subplots(1, 1, figsize=(9.5, 3))
    zeroes = distribution[:, 0]
    ones = distribution[:, 1]


    axs.bar(np.arange(len(zeroes)), zeroes, label="0", color=colors[3])
    axs.bar(np.arange(len(zeroes)), ones, bottom=zeroes, color=colors[0], label="1")
    axs.set_title("Distribution of binary features")
    axs.set_xticks(np.arange(len(d.binary_features)))
    axs.set_xticklabels(readable_feature_names, rotation=0)
    axs.legend()

    fig.tight_layout()
    plt.savefig("plots/binary_features.pdf")

def distribution_of_numerical_features():
    unique_counts = np.zeros((len(d.numeric_features)))
    for i, index in enumerate(d.numeric_features):
        feats = np.concatenate([d.c_train_x[:, index].squeeze().numpy(), d.c_test_x[:, index].squeeze().numpy(), d.e_x[:, index].squeeze().numpy()])
        unique_counts[i] = len(np.unique(feats))

    # Dirty hack: For some reason, the first value is not used for x axis label
    readable_feature_names = [""] + ["\n".join(name.split("_")) for name in np.array(d.feature_names)[d.numeric_features]]

    fig, axs = plt.subplots(1, 1, figsize=(4.5, 3))
    axs.bar(np.arange(len(d.numeric_features)), unique_counts, width=0.5, color=colors[0])
    axs.set_yscale("log")
    axs.set_ylim(0, 1000)
    axs.set_title("Unique values in numerical features")
    axs.set_xticklabels(readable_feature_names)
    axs.set_ylabel("Number of unique values")

    fig.tight_layout()
    plt.savefig("plots/numerical.pdf")

def distribution_of_expert_annotations():
    expert_annotations = d.annot.numpy()
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))

    parts = axs.violinplot(expert_annotations, showmedians=True)
    for pc in parts['bodies']:
        pc.set_color(colors[0])
        pc.set_alpha(1)
    parts['cmedians'].set_color("black")
    parts['cmedians'].set_linewidth(3)
    axs.axhline(color="black", linestyle="--", alpha=0.6)
    axs.set_xticks(np.arange(19)+1)
    axs.set_ylabel("Expert feature importance rating")
    axs.set_xticklabels([" ".join(name.split("_")) for name in d.feature_names], rotation=-90)

    fig.tight_layout()
    plt.savefig("plots/expert_ratings.pdf")

if __name__ == "__main__":
    distribution_of_binary_features()
    distribution_of_numerical_features()
    distribution_of_expert_annotations()