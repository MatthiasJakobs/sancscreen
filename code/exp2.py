# Compare different explainability methods on the new dataset

from operator import gt
import torch
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import re
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from metrics import L1, L2, generate_mean_ranking, spearman_footrule, shamming, scosine, seuclidean
from datasets import load_sancscreen
from model import SancScreenNet
from parameters import sancscreen_config
from attributions import DIY_EG

def plot_boxplots(all_attr, gt_attributions, all_labels, names):
    shadings = ["sandybrown", "violet", "salmon", "lightblue", "palegreen", "lightgrey"]
    edges = ["saddlebrown", "purple", "darkred", "blue", "green", "gray"]
    plt.rcParams['font.size'] = '12'
    fig, ax = plt.subplots(2, 3, sharey=True, figsize=(15, 6))
    for column, (metric_name, metric_fn) in enumerate(zip(["Spearmans Footrule", "Cosine distance", "Hamming distance"], [spearman_footrule, scosine, shamming])):
        for model in range(len(all_attr)):
            attr = all_attr[model]
            metric_results = np.zeros((len(attr), len(attr[0])))
            for method in range(len(attr)):
                e = attr[method]
                metric_results[method] = metric_fn(e, gt_attributions)

            sorted = np.argsort(np.median(metric_results, axis=1))
            for i in range(len(sorted)):
                j = sorted[i]
                flier = dict(marker="+", alpha=0.5)
                boxes = dict(color=edges[j], facecolor=shadings[j])
                whiskers = dict(color=edges[j])
                ms = dict(color="black")
                caps = dict(color=edges[j])

                ax[model, column].boxplot(metric_results[j], positions=[i], patch_artist=True, widths=0.25, flierprops=flier, boxprops=boxes, whiskerprops=whiskers, medianprops=ms, capprops=caps)

            #if column == 0:
                #ax[model, column].set_title(f"{names[model]}")
            ax[model, column].set_ylabel(metric_name)
            ax[model, column].set_ylim(0, 1)
            ax[model, column].set_title(f"{names[model]}")
            ax[model, column].set_xticklabels(np.array(all_labels[model])[sorted], rotation=20)

    plt.tight_layout()
    plt.savefig(f"plots/exp2.pdf")


def lime_explanation_to_numpy(lime_attr, feature_names):
    # Match ordering of string array a to b
    def get_ordering(a, b):
        ordering = np.zeros((len(a)))
        for j, s in enumerate(a):
            for i, z in enumerate(b):
                if s == z:
                    ordering[j] = i
                    continue
        return ordering.astype(np.int8)

    attr = np.zeros((len(lime_attr)))
    labels = [x[0] for x in lime_attr]
    values = [x[1] for x in lime_attr]

    cleaned_labels = [re.search(r'[a-zA-Z0-9]+_([a-zA-Z0-9]._?)+', x).group(0).replace(" ", "") for x in labels]
    ordering = get_ordering(cleaned_labels, feature_names)

    for i, ind in enumerate(ordering):
        attr[ind] = values[i]

    return attr.reshape(1, -1)

def run_neural_network(X_train, y_train, X_test, y_test, X, y, gt_attributions, feature_names, c):
    print("NEURAL NETWORK")
    model = SancScreenNet(c)
    model.load_state_dict(torch.load("results/checkpoints/sancscreen_net.pth"))

    background = shap.sample(X_train, 100, random_state=123)


    model_to_lime = lambda x: np.concatenate([(np.ones_like(x.shape[0]) - model.predict_proba_numpy(x)).reshape(-1, 1), model.predict_proba_numpy(x).reshape(-1, 1)], axis=1)

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=["STOP", "GO"], discretize_continuous=True)
    try:
        lime_attr = np.load("results/exp2/nn_lime.npy")
    except:
        lime_attr = np.concatenate([lime_explanation_to_numpy(explainer.explain_instance(x, model_to_lime, num_features=19).as_list(), feature_names) for x in X], axis=0)
        np.save("results/exp2/nn_lime.npy", lime_attr)

    deep_explainer = shap.DeepExplainer(model, torch.from_numpy(background))
    try:
        deep_attr = np.load("results/exp2/nn_deepshap.npy")
    except:
        deep_attr = deep_explainer.shap_values(torch.from_numpy(X))
        np.save("results/exp2/nn_deepshap.npy", deep_attr)

    ks_explainer = shap.KernelExplainer(lambda x: model(torch.from_numpy(x)).detach().numpy(), background)
    try:
        ks_attr = np.load("results/exp2/nn_ks_data.npy")
    except:
        ks_attr = ks_explainer.shap_values(X)[0]
        np.save("results/exp2/nn_ks_data.npy", ks_attr)

    ks_explainer = shap.KernelExplainer(lambda x: model(torch.from_numpy(x)).detach().numpy(), np.zeros((1, 19)).astype(np.float32))
    try:
        ks_zero_attr = np.load("results/exp2/nn_ks_zero.npy")
    except:
        ks_zero_attr = ks_explainer.shap_values(X)[0]
        np.save("results/exp2/nn_ks_zero.npy", ks_zero_attr)

    ig_explainer = IntegratedGradients(model)
    background = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(len(X), 1)
    try:
        ig_attr = np.load("results/exp2/nn_ig.npy")
    except:
        ig_attr = ig_explainer.attribute(torch.from_numpy(X), baselines=background, n_steps=100, return_convergence_delta=False)
        ig_attr = ig_attr.detach().numpy()
        np.save("results/exp2/nn_ig.npy", ig_attr)

    eg_explainer = DIY_EG(model, background=torch.from_numpy(X_train))
    try:
        eg_attr = np.load("results/exp2/nn_eg.npy")
    except:
        eg_attr = eg_explainer.attribute(torch.from_numpy(X), labels=None, k=100)
        eg_attr = eg_attr.detach().numpy()
        np.save("results/exp2/nn_eg.npy", eg_attr)

    nn_attributions = [lime_attr, ks_zero_attr, ks_attr, deep_attr, ig_attr, eg_attr]
    return ["LIME", "KS-Zero", "KS-Data", "DeepSHAP", "IG", "EG"], nn_attributions


def run_random_forest(X_train, y_train, X_test, y_test, X, y, gt_attributions, feature_names):

    # Fit model
    model = RandomForestClassifier(n_estimators=300, random_state=123)
    model.fit(X_train, y_train)

    print("RANDOM FOREST")

    # Experiment Log

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=["STOP", "GO"], discretize_continuous=True)
    try:
        lime_attr = np.load("results/exp2/rf_lime.npy")
    except:
        lime_attr = np.concatenate([lime_explanation_to_numpy(explainer.explain_instance(x, model.predict_proba, num_features=19).as_list(), feature_names) for x in X], axis=0)
        np.save("results/exp2/rf_lime.npy", lime_attr)

    background = shap.sample(X_train, 100, random_state=123)

    # # TreeSHAP
    tree_explainer = shap.TreeExplainer(model, background, model_output="probability")
    try:
        tree_int_attr = np.load("results/exp2/rf_tree_int.npy")
    except:
        tree_int_attr = tree_explainer.shap_values(X)[1]
        np.save("results/exp2/rf_tree_int.npy", tree_int_attr)

    tree_explainer = shap.TreeExplainer(model)
    try:
        tree_attr = np.load("results/exp2/rf_tree.npy")
    except:
        tree_attr = tree_explainer.shap_values(X)[1]
        np.save("results/exp2/rf_tree.npy", tree_attr)

    # # # KS (zero baseline)
    ks_explainer = shap.KernelExplainer(model.predict_proba, np.zeros((1, 19)))
    try:
        ks_zero_attr = np.load("results/exp2/rf_ks_zero.npy")
    except:
        ks_zero_attr = ks_explainer.shap_values(X)[1]
        np.save("results/exp2/rf_ks_zero.npy", ks_zero_attr)

    # # # # KS (dataset baseline)
    ks_explainer = shap.KernelExplainer(model.predict_proba, background)
    try:
        ks_data_attr = np.load("results/exp2/rf_ks_data.npy")
    except:
        ks_data_attr = ks_explainer.shap_values(X)[1]
        np.save("results/exp2/rf_ks_data.npy", ks_data_attr)

    rf_attributions = [lime_attr, ks_zero_attr, ks_data_attr, tree_attr, tree_int_attr, ]
    return ["LIME", "KS-Zero", "KS-Data", "TreeSHAP-Cond", "TreeSHAP-Int"], rf_attributions


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

    plot_boxplots([rf_attributions, nn_attributions], gt_attributions, [rf_labels, nn_labels], ["Random Forest", "Neural Network"])
