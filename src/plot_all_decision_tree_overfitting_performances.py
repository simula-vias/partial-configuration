# Analysis: Are the best configurations in our labels?

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from common import load_data, baseline_results_wc
import glob

# Load data for all systems
data_dir = Path("./data")
systems = json.load(open(data_dir / "metadata.json")).keys()
random_state = 1234

for s in systems:
    # Load data
    (
        perf_matrix_initial,
        input_features,
        config_features,
        all_performances,
        input_preprocessor,
        config_preprocessor,
    ) = load_data(system=s, data_dir=data_dir.absolute(), input_properties_type="tabular")

    performances = all_performances

    nmdf = (
        perf_matrix_initial[["inputname"] + performances]
        .groupby("inputname", as_index=True)
        # .transform(lambda x: scale(x))
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    )
    nmdf["worst_case_performance"] = nmdf[performances].max(axis=1)
    perf_matrix = pd.merge(
        perf_matrix_initial,
        nmdf,
        suffixes=("_raw", None),
        left_index=True,
        right_index=True,
    )
    perf_matrix["rank"] = perf_matrix.groupby("inputname", group_keys=False).apply(
        lambda x: x["worst_case_performance"].argsort() + 1, include_groups=False
    )
    # We adjust the WCP by expressing it as the difference from the best WCP, i.e. the best WCP is always 0
    perf_matrix["worst_case_performance"] = (
        perf_matrix[["inputname", "worst_case_performance"]]
        .groupby("inputname", as_index=True)
        .transform(lambda x: x - x.min())
    )
    dataset = (
        perf_matrix.groupby("inputname")
        .apply(
            lambda x: x.loc[x["worst_case_performance"].idxmin()], include_groups=False
        )
        .reset_index()
    )

    splits = 5
    kf_inp = KFold(n_splits=splits, random_state=None, shuffle=False)

    inputnames = perf_matrix["inputname"].unique()

    # Store results for all splits
    all_train_acc = []
    all_train_acc2 = []
    all_test_acc = []
    all_test_acc2 = []
    all_wcp = []
    all_wcp2 = []

    ## Baselines
    for split_idx, (train_inp_idx, test_inp_idx) in enumerate(kf_inp.split(inputnames)):
        train_inp = sorted(inputnames[train_inp_idx])
        test_inp = sorted(inputnames[test_inp_idx])
        train_perf = dataset[dataset.inputname.isin(train_inp)]
        test_perf = dataset[dataset.inputname.isin(test_inp)]

        icm = (
            perf_matrix[perf_matrix.inputname.isin(train_inp)][
                ["inputname", "configurationID", "worst_case_performance"]
            ]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
        )
        icm_all_perf = (
            perf_matrix[["inputname", "configurationID"] + all_performances]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
        )

        dataset = icm.join(config_features).join(input_features).reset_index()

        icm_test = (
            perf_matrix[~perf_matrix.inputname.isin(train_inp)][
                ["inputname", "configurationID", "worst_case_performance"]
            ]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
        )
        baseline_results = baseline_results_wc(
            icm,
            icm_all_perf,
            icm_test,
            dataset,
            config_features,
        )

        def eval_prediction(pred_cfg_test):
            inp_pred_map = pd.DataFrame(
                zip(test_inp, pred_cfg_test), columns=["inputname", "configurationID"]
            )
            return perf_matrix.merge(inp_pred_map, on=["inputname", "configurationID"])[
                "worst_case_performance"
            ].mean()

        target_preprocessor = LabelEncoder()
        target_preprocessor.fit(dataset["configurationID"])

        X = input_preprocessor.fit_transform(train_perf[input_features.columns])
        y = target_preprocessor.transform(train_perf["configurationID"])

        X_test = input_preprocessor.transform(test_perf[input_features.columns])
        y_test = target_preprocessor.transform(test_perf["configurationID"])

        X_all = np.concatenate((X, X_test))
        y_all = np.concatenate((y, y_test))

        n_trees = list(range(2, 17, 2)) # + [100]  # From 2 to 16 in steps of 2
        train_acc = []
        train_acc2 = []
        test_acc = []
        test_acc2 = []
        wcp = []
        wcp2 = []

        for n_est in n_trees:
            clf = RandomForestClassifier(
                n_estimators=n_est,
                criterion="entropy",
                max_depth=None,
                random_state=random_state
            )
            clf = clf.fit(X, y)
            pred_cfg_test = target_preprocessor.inverse_transform(clf.predict(X_test))

            clf2 = RandomForestClassifier(
                n_estimators=n_est,
                criterion="entropy",
                max_depth=None,
                random_state=random_state
            )
            clf2 = clf2.fit(X_all, y_all)
            pred_cfg_test2 = target_preprocessor.inverse_transform(clf2.predict(X_test))

            train_acc.append(clf.score(X, y))
            train_acc2.append(clf2.score(X_all, y_all))
            test_acc.append(clf.score(X_test, y_test))
            test_acc2.append(clf2.score(X_test, y_test))
            wcp.append(eval_prediction(pred_cfg_test))
            wcp2.append(eval_prediction(pred_cfg_test2))

        all_train_acc.append(train_acc)
        all_train_acc2.append(train_acc2)
        all_test_acc.append(test_acc)
        all_test_acc2.append(test_acc2)
        all_wcp.append(wcp)
        all_wcp2.append(wcp2)

    mean_train_acc = np.mean(all_train_acc, axis=0)
    std_train_acc = np.std(all_train_acc, axis=0)
    mean_train_acc2 = np.mean(all_train_acc2, axis=0)
    std_train_acc2 = np.std(all_train_acc2, axis=0)
    mean_test_acc = np.mean(all_test_acc, axis=0)
    std_test_acc = np.std(all_test_acc, axis=0)
    mean_test_acc2 = np.mean(all_test_acc2, axis=0)
    std_test_acc2 = np.std(all_test_acc2, axis=0)
    mean_wcp = np.mean(all_wcp, axis=0)
    std_wcp = np.std(all_wcp, axis=0)
    mean_wcp2 = np.mean(all_wcp2, axis=0)
    std_wcp2 = np.std(all_wcp2, axis=0)

    best_wcp = mean_wcp.min() <= baseline_results["overall_avg"]
    best_wcp2 = mean_wcp2.min() <= baseline_results["overall_avg"]

    print(f"Best WCP: {best_wcp}, Best WCP2: {best_wcp2}")

    # Plotting for each system
    plt.figure(figsize=(10, 6))
    
    colors = {
        'train': 'blue',
        'test': 'red',
        'wcp': 'green'
    }
    
    plt.plot(n_trees, mean_train_acc, color=colors['train'], label="Train Accuracy", linestyle='-')
    plt.fill_between(n_trees, mean_train_acc - std_train_acc,
                    mean_train_acc + std_train_acc, color=colors['train'], alpha=0.2)
    
    plt.plot(n_trees, mean_test_acc, color=colors['test'], label="Test Accuracy", linestyle='-')
    plt.fill_between(n_trees, mean_test_acc - std_test_acc,
                    mean_test_acc + std_test_acc, color=colors['test'], alpha=0.2)
    
    plt.plot(n_trees, mean_wcp, color=colors['wcp'], label="WCP", linestyle='-')
    plt.fill_between(n_trees, mean_wcp - std_wcp,
                    mean_wcp + std_wcp, color=colors['wcp'], alpha=0.2)
    
    plt.plot(n_trees, mean_train_acc2, color=colors['train'], 
            label="Train Accuracy (Trained on All Data)", linestyle='--')
    plt.fill_between(n_trees, mean_train_acc2 - std_train_acc2,
                    mean_train_acc2 + std_train_acc2, color=colors['train'], alpha=0.2)
    
    plt.plot(n_trees, mean_test_acc2, color=colors['test'],
            label="Test Accuracy (Trained on All Data)", linestyle='--')
    plt.fill_between(n_trees, mean_test_acc2 - std_test_acc2,
                    mean_test_acc2 + std_test_acc2, color=colors['test'], alpha=0.2)
    
    plt.plot(n_trees, mean_wcp2, color=colors['wcp'],
            label="WCP (Trained on All Data)", linestyle='--')
    plt.fill_between(n_trees, mean_wcp2 - std_wcp2,
                    mean_wcp2 + std_wcp2, color=colors['wcp'], alpha=0.2)

    plt.xlabel("Number of Trees")
    plt.ylabel("Score")
    plt.legend()
    plt.title(f"{s}, {splits}-fold CV")
    plt.grid(True)
    plt.savefig(data_dir / f"../plots/random_forest_overfitting_{s}.png")
    plt.close()

# After processing all systems, create a grid of plots
plot_files = sorted(glob.glob(str(data_dir / "../plots/random_forest_overfitting_*.png")))
n_plots = len(plot_files)
n_rows = 2
n_cols = (n_plots + n_rows - 1) // n_rows

# Create figure for the grid
fig = plt.figure(figsize=(15, 6))

# Add each plot as a subplot
for i, plot_file in enumerate(plot_files):
    img = plt.imread(plot_file)
    ax = fig.add_subplot(n_rows, n_cols, i+1)
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.savefig(data_dir / '../plots/random_forest_overfitting_grid.png', bbox_inches='tight', dpi=400)
plt.close()
