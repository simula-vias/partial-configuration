import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from common import label_inputs_greedy, load_data
from odt_labelled import prepare_perf_matrx, binarize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

metadata = json.load(open("./data/metadata.json"))
all_bin_sizes = [5, 8, 10, 12, 15, 20]
all_res = []
for system in metadata.keys():
    res = []
    for num_bins_feature_encoding in all_bin_sizes:
        (
            perf_matrix_initial,
            input_features,
            config_features,
            all_performances,
            input_preprocessor,
            config_preprocessor,
        ) = load_data(
            system=system,
            data_dir="./data",
            input_properties_type="tabular",
            num_bins_feature_encoding=num_bins_feature_encoding,
        )

        perf_matrix = prepare_perf_matrx(perf_matrix_initial, all_performances)

        C = (
            perf_matrix[["inputname", "configurationID", "worst_case_performance"]]
            .reset_index()
            .pivot(
                index="inputname",
                columns="configurationID",
                values="worst_case_performance",
            )
            .sort_values("inputname")
        )
        # Label each input with the configuration with the lowest WCP for reference
        # y_argmin = np.argmin(C, axis=1)
        y_argmin_idx = label_inputs_greedy(perf_matrix, 5)
        y_argmin = perf_matrix.configurationID.unique()[y_argmin_idx]

        enc = LabelEncoder()
        y = enc.fit_transform(y_argmin)

        input_feature_columns = input_features.columns
        feature_matrix = perf_matrix[
            list(input_feature_columns) + ["inputname"]
        ].drop_duplicates()
        features = feature_matrix[input_feature_columns]
        inputnames = feature_matrix["inputname"]

        train_inp_idx, test_inp_idx = train_test_split(
            np.arange(len(inputnames)), test_size=0.2, random_state=42
        )
        train_inp = sorted(inputnames.iloc[train_inp_idx])
        test_inp = sorted(inputnames.iloc[test_inp_idx])

        icm_test = (
            perf_matrix[~perf_matrix.inputname.isin(train_inp)][
                ["inputname", "configurationID", "worst_case_performance"]
            ]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
        )

        # Transform features
        X = input_preprocessor.fit_transform(features)
        feature_names = input_preprocessor.get_feature_names_out()

        X_train = X[train_inp_idx]
        X_test = X[test_inp_idx]
        y_train = y[train_inp_idx]
        y_test = y[test_inp_idx]

        def eval_prediction(pred_cfg_test):
            inp_pred_map = pd.DataFrame(
                zip(test_inp, pred_cfg_test),
                columns=["inputname", "configurationID"],
            )
            return icm_test.merge(inp_pred_map, on=["inputname", "configurationID"])[
                "worst_case_performance"
            ]  # .mean()

        for d in [2, 3, 4, 5, 7, 10]:
            clf = DecisionTreeClassifier(max_depth=d)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            pred_cfg = enc.inverse_transform(y_pred).astype(int)
            eval_pred = eval_prediction(pred_cfg).mean()
            classes_used = np.unique(y_argmin).shape[0]
            accuracy = accuracy_score(y_test, y_pred)
            num_features = X.shape[1]
            print(
                f"{system}@{d}\t{num_bins_feature_encoding} ({classes_used})\t{num_features}\t{accuracy:.2%}\t{eval_pred:.3f}"
            )
            all_res.append(
                (
                    system,
                    d,
                    num_bins_feature_encoding,
                    eval_pred,
                    classes_used,
                    accuracy,
                    num_features,
                )
            )
            res.append(eval_prediction(pred_cfg).mean())

all_res = pd.DataFrame(
    all_res,
    columns=[
        "system",
        "depth",
        "num_bins_feature_encoding",
        "eval_pred",
        "classes_used",
        "accuracy",
        "num_features",
    ],
)
all_res.to_csv("odt_num_features.csv", index=False)

# Analysis; best num_bins_feature_encoding for each system on top
df = pd.read_csv("../odt_num_features.csv")
df.groupby(["system", "num_bins_feature_encoding"]).agg(
    {"eval_pred": "mean", "accuracy": "mean", "num_features": "mean"}
).round(3).sort_values(
    ["system", "eval_pred", "accuracy"], ascending=[True, True, False]
)
