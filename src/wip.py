# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from odtlearn.flow_oct import FlowOCT
from odtlearn.utils.binarize import binarize

from common import load_data
import json

random_state = 1234
test_size = 0.40

system = "gcc"
performance_str = "['exec']"
num_configs = 2
performances = json.loads(performance_str.replace("'", '"'))

best_configs = json.load(open("../results/ocs_results.json"))

res = best_configs[system][performance_str][str(num_configs)]
selected_configs = res["selected_configs"]

(
    perf_matrix_initial,
    input_features,
    config_features,
    all_performances,
    input_preprocessor,
    config_preprocessor,
) = load_data(system=system, data_dir="../data", input_properties_type="tabular")

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
# We adjust the WCP by expressing it as the difference from the best WCP, i.e. the best WCP is always 0
perf_matrix["worst_case_performance"] = (
    perf_matrix[["inputname", "worst_case_performance"]]
    .groupby("inputname", as_index=True)
    .transform(lambda x: x - x.min())
)

# Split data
train_inp, test_inp = train_test_split(
    perf_matrix["inputname"].unique(),
    test_size=test_size,
    random_state=random_state,
)

# This is only to make our evaluation simpler
train_inp = sorted(train_inp)
test_inp = sorted(test_inp)

train_perf = perf_matrix[perf_matrix.inputname.isin(train_inp)].copy()
test_perf = perf_matrix[perf_matrix.inputname.isin(test_inp)]

features = train_perf[input_features.columns].drop_duplicates()

# Transform features
# X = input_preprocessor.fit_transform(features)
# feature_names = input_preprocessor.get_feature_names_out()

# Create cost matrix for evaluation
C = (
    perf_matrix[perf_matrix.inputname.isin(train_inp)][
        ["inputname", "configurationID", "worst_case_performance"]
    ]
    .reset_index()
    .pivot(
        index="inputname", columns="configurationID", values="worst_case_performance"
    )
    .sort_values("inputname")
    .reset_index()
    .drop(columns=["inputname"])
    .values
)
# Label each input with the configuration with the lowest WCP
y = np.argmin(C, axis=1)

# Print class distribution
print("Class distribution:")
unique_classes, class_counts = np.unique(y, return_counts=True)
for c, count in zip(unique_classes, class_counts):
    print(f"Class {c}: {count} ({100 * count / len(y):.2f}%)")

# print(X.head())

X = binarize(
    features,
    categorical_cols=[],
    integer_cols=features.columns,
    real_cols=[],
    n_bins=5,
)

# print(X.head())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

stcl = FlowOCT(
    depth=2,
    solver="cbc",
    time_limit=100,
)

stcl.fit(X_train, y_train)
stcl.print_tree()
y_pred = stcl.predict(X_train)
test_pred = stcl.predict(X_test)
print(
    "The out-of-sample acc is {}".format(np.sum(test_pred == y_test) / y_test.shape[0])
)

def evaluate_cost(y_pred, C):
    total_cost = 0
    for i, pred in enumerate(y_pred):
        total_cost += C[i, pred]
    return total_cost / len(y_pred)


print(y_pred, evaluate_cost(y_pred, C))
print(test_pred, evaluate_cost(test_pred, C))

# %%
