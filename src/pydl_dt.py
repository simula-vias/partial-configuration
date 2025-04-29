# %%
import numpy as np
from sklearn.model_selection import train_test_split
from gosdt import ThresholdGuessBinarizer, GOSDTClassifier, NumericBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from common import load_data
import json
import pandas as pd

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
X = input_preprocessor.fit_transform(features)
feature_names = input_preprocessor.get_feature_names_out()

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

# dataset = np.genfromtext("anneal.txt", delimiter=" ")

# discretizer = KBinsDiscretizer(n_bins=3, encode="onehot")
# X_bin = discretizer.fit_transform(X).toarray()
X_bin = X
# %%

# Parameters
GBDT_N_EST = 4
GBDT_MAX_DEPTH = 40
REGULARIZATION = 0.00  # 001
SIMILAR_SUPPORT = False
DEPTH_BUDGET = 60
TIME_LIMIT = 3600
VERBOSE = True

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_bin, y, test_size=0.2, random_state=2021
)
print("X train shape:{}, X test shape:{}".format(X_train.shape, X_test.shape))

# Step 1: Guess Thresholds
X_train = pd.DataFrame(X_train)  # , columns=feature_names)
X_test = pd.DataFrame(X_test)  # , columns=feature_names)

print("Original feature names:", list(X_train.columns))
print("Number of original features:", X_train.shape[1])


# enc = ThresholdGuessBinarizer(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH, random_state=2021)
# enc = NumericBinarizer()
# enc.set_output(transform="pandas")
# X_train_guessed = enc.fit_transform(X_train, y_train)
# X_test_guessed = enc.transform(X_test)
X_train_guessed = X_train.copy()
X_test_guessed = X_test.copy()

print(
    f"After guessing, X train shape:{X_train_guessed.shape}, X test shape:{X_test_guessed.shape}"
)
print(
    "train set column names == test set column names: {list(X_train_guessed.columns)==list(X_test_guessed.columns)}"
)
print("Transformed feature names:", list(X_train_guessed.columns))
print("Number of transformed features:", X_train_guessed.shape[1])

# Check for unique values in transformed features
for col in X_train_guessed.columns:
    unique_values = X_train_guessed[col].unique()
    print(f"Unique values in {col}: {unique_values}")
    if len(unique_values) > 2:
        print(f"WARNING: Feature {col} has more than 2 unique values.")

classes = np.unique(y_train)
class_map = np.arange(len(classes))
print("Classes:", classes)
print("Class map:", class_map)

y_train = np.array([class_map[np.where(classes == c)[0][0]] for c in y_train])
# y_test = np.array([class_map[np.where(classes == c)[0][0]] for c in y_test])

# Step 3: Train the GOSDT classifier
clf = GOSDTClassifier(
    regularization=1/y_train.shape[0],
    similar_support=SIMILAR_SUPPORT,
    time_limit=TIME_LIMIT,
    depth_budget=DEPTH_BUDGET,
    verbose=VERBOSE,
)
clf.fit(X_train_guessed, y_train)
y_pred = clf.predict(X_train_guessed)

y_pred = class_map[y_pred]

# Step 4: Evaluate the model
print("Evaluating the model, extracting tree and scores", flush=True)


print(f"Model training time: {clf.result_.time}")
print(f"Training accuracy: {clf.score(X_train_guessed, y_train)}")
# print(f"Test accuracy: {clf.score(X_test_guessed, y_test)}")

clf_sk = DecisionTreeClassifier(max_depth=1)
clf_sk.fit(X_bin, y)
y_pred_sk = clf_sk.predict(X_bin)

def evaluate_cost(y_pred, C):
    total_cost = 0
    for i, pred in enumerate(y_pred):
        total_cost += C[i, pred]
    return total_cost / len(y_pred)


print(y_pred, evaluate_cost(y_pred, C))
print(y_pred_sk, evaluate_cost(y_pred_sk, C))

# %%
from odtlearn.flow_oct import FlowOCT
from odtlearn.utils.binarize import binarize


# %%
