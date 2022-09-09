# Compare correlation on a global scale
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd

from metrics import spearman_footrule, shamming, scosine
from datasets import load_sancscreen
from parameters import sancscreen_config
from scipy.stats import wilcoxon, pearsonr

from exp2 import run_random_forest, run_neural_network

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

    df = pd.DataFrame()

    for model_name, attr, labels in zip(["RF", "NN"], [rf_attributions, nn_attributions], [rf_labels, nn_labels]):
        for idx, exp_method_label in enumerate(labels):
            rs = []
            ps = []
            for feat in range(19):
                e = attr[idx][:, feat].squeeze()
                if len(np.unique(e)) == 1:
                    print(f"Attribution for {feat} is constant, add small noise value")
                    e = e + rng.normal(size=len(e)) * 1e-5

                r, p = pearsonr(e, gt_attributions[:, feat].squeeze())
                rs.append(r)
                ps.append(p)
            df = df.append({"model": model_name, "exp": exp_method_label, "r": rs, "p": ps}, ignore_index=True)

    heatmap = np.zeros((11, 19))
    for i, row in df.iterrows():
        heatmap[i] = np.array(row[3])

    y_labels = [f"{m} {e}" for m, e in zip(df["model"].tolist(), df["exp"].tolist())]

    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    fig, ax = plt.subplots(1, 1, figsize=(5.3, 3.5))
    im = ax.imshow(heatmap, cmap=plt.get_cmap("bwr"), norm=norm)
    ax.axhline(4.5, color="black")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(19))
    ax.set_xticklabels([" ".join(f.split("_")) for f in d.feature_names], rotation=-90)

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    plt.colorbar(im, cax=cax)

    fig.tight_layout()
    plt.savefig("plots/correlation.pdf")
