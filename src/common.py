import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import _MultiOutputEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import ClassifierMixin


def find_files(path, ext="csv"):
    ext = ext
    # list all the files in the folder
    filenames = os.listdir(path)
    # list files that have the specified extension
    filename_list = [filename for filename in filenames if filename.endswith(ext)]
    return filename_list


def load_csv(path, filename, with_name=False):
    filename = os.path.join(path, filename)
    data = pd.read_csv(filename, header=0)

    if with_name:
        data["inputname"] = os.path.splitext(os.path.basename(filename))[0]

    nb_data = data.shape[0]
    return data, nb_data


def load_all_csv(path, ext="csv", with_names=False):
    files_to_load = find_files(path, ext)
    # load first data file alone
    all_data, nb_config = load_csv(path, files_to_load[0], with_name=with_names)
    # load the rest and append to the previous dataframe
    for f in files_to_load[1:]:
        app_data, a = load_csv(path, f, with_name=with_names)
        all_data = pd.concat([all_data, app_data])
    # all_data = pd.concat([pd.read_csv(path+'/'+f) for f in files_to_load])

    return all_data, nb_config


def load_data(system, input_properties_type="tabular", data_dir="../data"):
    if input_properties_type == "embedding" and system not in ("gcc",):
        raise NotImplementedError(
            f"Input properties `embedding` only available for (gcc,), not `{system}`"
        )

    metadata = json.load(open(os.path.join(data_dir, "metadata.json")))
    system_metadata = metadata[system]
    config_columns = system_metadata["config_columns"]
    config_columns_cat = system_metadata["config_columns_cat"]
    config_columns_cont = system_metadata["config_columns_cont"]
    # input_columns = system_metadata["input_columns"]
    input_columns_cat = system_metadata["input_columns_cat"]
    input_columns_cont = system_metadata["input_columns_cont"]
    performances = system_metadata["performances"]

    meas_matrix, _ = load_all_csv(
        os.path.join(data_dir, system), ext="csv", with_names=True
    )

    if system == "nodejs":
        # nodejs is missing the configurationID in the measurements
        # We re-assign it by the line number of the measurement in the resp. file
        meas_matrix["configurationID"] = meas_matrix.index.rename("configurationID")

    if input_properties_type == "embedding":
        input_properties_file = "input_embeddings.csv"
        input_columns_cat = []
        input_columns_cont = [f"v{i}" for i in range(768)]
    else:
        input_properties_file = "properties.csv"

    input_properties = pd.read_csv(
        os.path.join(data_dir, system, "others", input_properties_file),
        dtype={"name": "object"},
    ).set_index("id")  # id needed?

    # Rename columns with same name in inputs/perf. prediction to avoid errors later
    # Affects imagemagick and xz
    for c in input_properties.columns:
        if c in performances or c in config_columns:
            new_col_name = f"inp_{c}"
            input_properties.rename(columns={c: new_col_name}, inplace=True)

            if c in input_columns_cat:
                input_columns_cat.remove(c)
                input_columns_cat.append(new_col_name)
            elif c in input_columns_cont:
                input_columns_cont.remove(c)
                input_columns_cont.append(new_col_name)

    perf_matrix = pd.merge(
        meas_matrix, input_properties, left_on="inputname", right_on="name"
    ).sort_values(by=["inputname", "configurationID"])
    del perf_matrix["name"]

    inputs_before_filter = len(perf_matrix.inputname.unique())
    configs_before_filter = len(perf_matrix.configurationID.unique())
    assert (
        inputs_before_filter * configs_before_filter == perf_matrix.shape[0]
    ), "Num. inputs * num. configs does not match measurement matrix before filtering"

    # System-specific adjustments
    if system == "gcc":
        # size=0 outputs in gcc seem to be invalid
        perf_matrix = perf_matrix[
            (
                perf_matrix[["inputname", "size"]].groupby("inputname").transform("min")
                > 0
            ).values
        ]
    elif system == "lingeling":
        # cps=0 outputs in lingeling seem to be invalid
        perf_matrix = perf_matrix[
            (
                perf_matrix[["inputname", "cps"]].groupby("inputname").transform("min")
                > 0
            ).values
        ]
    elif system == "nodejs":
        perf_matrix["ops"] = -perf_matrix[
            "ops"
        ]  # ops is the only increasing performance measure
    elif system == "x264":
        # perf_matrix["rel_size"] = perf_matrix["size"] / perf_matrix["ORIG_SIZE"]  # We have `kbs` which is a better alternative
        # perf_matrix["rel_size"] = np.log(perf_matrix["rel_size"])  # To scale value distribution more evenly
        perf_matrix["rel_kbs"] = perf_matrix["kbs"] / perf_matrix["ORIG_BITRATE"]

        # For fps and kbs higher is better
        perf_matrix["fps"] = -perf_matrix[
            "fps"
        ]
        perf_matrix["kbs"] = -perf_matrix[
            "kbs"
        ]

    # Drop inputs with constant measurements
    perf_matrix = perf_matrix[
        (
            perf_matrix[["inputname"] + performances]
            .groupby("inputname")
            .transform("std")
            > 0
        ).all(axis=1)
    ]

    # Correlation of performances
    # 1. Drops performances with perfect correlation
    # 2. Sorts performances by increasing avg. correlation
    perf_correlations = perf_matrix[performances].corr().drop_duplicates()
    performances = perf_correlations.mean(axis=1).sort_values().index.tolist()

    inputs_after_filter = len(perf_matrix.inputname.unique())
    configs_after_filter = len(perf_matrix.configurationID.unique())
    # print(
    #     f"Removed {inputs_before_filter-inputs_after_filter} inputs and {configs_before_filter-configs_after_filter} configs"
    # )
    assert (
        inputs_after_filter * configs_after_filter == perf_matrix.shape[0]
    ), "Num. inputs * num. configs does not match measurement matrix after filtering"

    # Separate input + config features
    input_features = (
        perf_matrix[["inputname"] + input_columns_cont + input_columns_cat]
        .drop_duplicates()
        .set_index("inputname")
    )

    config_features = (
        perf_matrix[["configurationID"] + config_columns_cont + config_columns_cat]
        .drop_duplicates()
        .set_index("configurationID")
    )

    # Prepare preprocessors, to be applied after data splitting
    if input_properties_type == "embedding":
        # Input embeddings are already scaled
        input_columns_cont = []

    input_preprocessor = ColumnTransformer(
        transformers=[
            # ("num", StandardScaler(), input_columns_cont),
            (
                "cat",
                OneHotEncoder(
                    min_frequency=1,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
                input_columns_cat,
            ),
        ],
        remainder="passthrough",
    )
    config_preprocessor = ColumnTransformer(
        transformers=[
            # ("num", StandardScaler(), config_columns_cont),
            (
                "cat",
                OneHotEncoder(
                    min_frequency=1,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
                config_columns_cat,
            ),
        ],
        remainder="passthrough",
    )

    return (
        perf_matrix,
        input_features,
        config_features,
        performances,
        input_preprocessor,
        config_preprocessor,
    )


def split_data(perf_matrix, test_size=0.2, verbose=True, random_state=None):
    # We set aside 20% of configurations and 20% of inputs as test data
    # This gives us 4 sets of data, of which we set 3 aside for testing
    train_cfg, test_cfg = train_test_split(
        perf_matrix["configurationID"].unique(),
        test_size=test_size,
        random_state=random_state,
    )
    train_inp, test_inp = train_test_split(
        perf_matrix["inputname"].unique(),
        test_size=test_size,
        random_state=random_state,
    )
    return make_split(
        perf_matrix, train_cfg, test_cfg, train_inp, test_inp, verbose=verbose
    )


def split_data_cv(perf_matrix, splits=4, verbose=True, random_state=None):
    kf_inp = KFold(n_splits=splits, random_state=random_state, shuffle=True)
    kf_cfg = KFold(n_splits=splits, random_state=random_state, shuffle=True)

    configuration_ids = perf_matrix["configurationID"].unique()
    inputnames = perf_matrix["inputname"].unique()

    for split_idx, (
        (train_cfg_idx, test_cfg_idx),
        (train_inp_idx, test_inp_idx),
    ) in enumerate(zip(kf_inp.split(configuration_ids), kf_cfg.split(inputnames))):
        train_cfg = configuration_ids[train_cfg_idx]
        test_cfg = configuration_ids[test_cfg_idx]
        train_inp = inputnames[train_inp_idx]
        test_inp = inputnames[test_inp_idx]

        split_dict = make_split(
            perf_matrix, train_cfg, test_cfg, train_inp, test_inp, verbose=verbose
        )
        split_dict["split"] = split_idx
        yield split_dict


def make_split(perf_matrix, train_cfg, test_cfg, train_inp, test_inp, verbose=True):
    train_cfg.sort()
    test_cfg.sort()
    train_inp.sort()
    test_inp.sort()
    train_data = perf_matrix[
        perf_matrix["configurationID"].isin(train_cfg)
        & perf_matrix["inputname"].isin(train_inp)
    ]

    test_data = perf_matrix[
        perf_matrix["configurationID"].isin(test_cfg)
        | perf_matrix["inputname"].isin(test_inp)
    ]
    test_cfg_new = perf_matrix[
        perf_matrix["configurationID"].isin(test_cfg)
        & perf_matrix["inputname"].isin(train_inp)
    ]
    test_inp_new = perf_matrix[
        perf_matrix["configurationID"].isin(train_cfg)
        & perf_matrix["inputname"].isin(test_inp)
    ]
    test_both_new = perf_matrix[
        perf_matrix["configurationID"].isin(test_cfg)
        & perf_matrix["inputname"].isin(test_inp)
    ]
    assert (
        test_cfg_new.shape[0]
        + test_inp_new.shape[0]
        + test_both_new.shape[0]
        + train_data.shape[0]
        == perf_matrix.shape[0]
    )

    if verbose:
        print(f"Training data: {100*train_data.shape[0]/perf_matrix.shape[0]:.2f}%")
        print(f"Both new: {100*test_both_new.shape[0]/perf_matrix.shape[0]:.2f}%")
        print(f"Config new: {100*test_cfg_new.shape[0]/perf_matrix.shape[0]:.2f}%")
        print(f"Input new: {100*test_inp_new.shape[0]/perf_matrix.shape[0]:.2f}%")

    return {
        "train_cfg": train_cfg,
        "test_cfg": test_cfg,
        "train_inp": train_inp,
        "test_inp": test_inp,
        "train_data": train_data,
        "test_data": test_data,
        # "test_data_cfg_new": test_cfg_new,
        # "test_data_inp_new": test_inp_new,
        # "test_data_both_new": test_both_new,
    }


## Rank-based distance calculation via correlation


# Function to calculate Pearson correlation coefficient in a vectorized manner
def pearson_correlation(X, Y):
    mean_X = np.mean(X, axis=-1, keepdims=True)
    mean_Y = np.mean(Y, axis=-1, keepdims=True)
    numerator = np.sum((X - mean_X) * (Y - mean_Y), axis=-1)
    denominator = np.maximum(
        np.sqrt(
            np.sum((X - mean_X) ** 2, axis=-1) * np.sum((Y - mean_Y) ** 2, axis=-1)
        ),
        1e-6,
    )
    return numerator / denominator


def pearson_rank_distance_matrix(measurements):
    # Vectorized spearson rank distance with multiple measurements

    # Breaks ties correctly by assigning same ranks, but potentially instable
    # TODO Should work correctly if we drop `frames` which has constant value
    ranks = stats.rankdata(measurements, axis=1, method="min")

    # Makes individual ranks
    # ranks = np.argsort(measurements, axis=1)

    # The ranks array is 3D (A, B, C), and we need to expand it to 4D for pairwise comparison in A,
    # while keeping C
    expanded_rank_X_3d = ranks[:, np.newaxis, :, :]  # Expanding for A dimension
    expanded_rank_Y_3d = ranks[np.newaxis, :, :, :]  # Expanding for A dimension

    A = ranks.shape[0]
    C = ranks.shape[2]

    # Initialize the Spearman correlation matrix for each C
    spearman_correlation_matrix_3d = np.empty((A, A, C))

    # Calculate Spearman correlation matrix for each C
    for c in range(C):
        spearman_correlation_matrix_3d[:, :, c] = pearson_correlation(
            expanded_rank_X_3d[:, :, :, c], expanded_rank_Y_3d[:, :, :, c]
        )

    return 1 - np.abs(spearman_correlation_matrix_3d)


def rank_difference_distance(measurements):
    ranks = np.argsort(measurements, axis=1)
    expanded_ranks = ranks[:, np.newaxis, :, :] - ranks[np.newaxis, :, :, :]

    # Calculate the absolute differences and sum along the B dimension
    vectorized_distance_matrix = np.sum(np.abs(expanded_ranks), axis=2)
    return vectorized_distance_matrix


def stat_distance(measurements, stats_fn):
    ranks = np.argsort(measurements, axis=1)
    A = ranks.shape[0]
    C = ranks.shape[2]

    distance_matrix = np.zeros((A, A, C))

    # There is no good vectorized version to apply,
    # therefore we loop over all dimensions...
    for c in range(C):
        for i in range(A):
            for j in range(i + 1, A):
                try:
                    res = stats_fn(ranks[i, :, c], ranks[j, :, c])
                    stat, _ = res.statistic, res.pvalue

                    distance_matrix[i, j, c] = stat
                    distance_matrix[j, i, c] = stat
                except ValueError:
                    # Mark as NaN in case of any other ValueError
                    distance_matrix[i, j, c] = np.nan
                    distance_matrix[j, i, c] = np.nan

    return distance_matrix


def kendalltau_distance(measurements):
    # We clip negative values to 0, because we see them as different
    # We invert such that lower values indicate higher correspondence
    return 1 - (np.maximum(stat_distance(measurements, stats_fn=stats.kendalltau), 0))


def wilcoxon_distance(measurements):
    return stat_distance(measurements, stats_fn=stats.wilcoxon)


def pareto_rank_numpy(data, cutoff=None):
    """Calculate the pareto front rank for each row in the numpy array."""
    unassigned = np.ones(len(data), dtype=bool)
    ranks = np.zeros(len(data), dtype=np.int32)
    front = 0
    
    if cutoff is not None:
        infeasible = (data > cutoff).any(axis=-1)
        unassigned[infeasible] = False

    while np.any(unassigned):
        front += 1

        # Adapted from https://github.com/QUVA-Lab/artemis/blob/peter/artemis/general/pareto_efficiency.py
        is_efficient = np.ones(data.shape[0], dtype=bool)
        is_efficient[~unassigned] = False
        for i, c in enumerate(data):
            if is_efficient[i]:
                # Keep any point with a lower cost or all cost exactly equal (for ties)
                is_efficient[is_efficient] = np.logical_or(
                    np.any(data[is_efficient] < c, axis=1),
                    np.all(data[is_efficient] == c, axis=1),
                )
                # is_efficient[i] = True  # And keep self

        ranks[is_efficient] = front
        unassigned[is_efficient] = False

    if cutoff is not None:
        # ranks[infeasible] = ranks.max() + 1
        ranks[infeasible] = len(data) + 1

    return ranks


def fonseca_fleming_rank(data, cutoff=None):
    rank = np.zeros(data.shape[0])  # initialize ranks

    # We make a n x n matrix and check for strictly smaller data on all measures
    dominated_counts = (data < data[:, None]).all(axis=-1).sum(axis=1)

    # +1 because the first rank is 1 (rank(x_i)  = 1 + p_i)
    # p_i the number of items by which x_i is dominated
    rank = dominated_counts + 1

    if cutoff is not None:
        infeasible = (data > cutoff).any(axis=-1)
        rank[infeasible] = rank.max() + 1

    return rank


def pareto_rank(pd_group, cutoff=None, rank_by_domination_count=True):
    if rank_by_domination_count:
        rank_fn = fonseca_fleming_rank
    else:
        rank_fn = pareto_rank_numpy

    return pd.Series(
        rank_fn(
            pd_group.values,
            cutoff=cutoff,
        ),
        index=pd_group.index,
    )


def baseline_results(
    icm,
    icm_ranked_measures,
    icm_test,
    dataset,
    config_features,
    verbose=False,
):
    ## These are our evaluation baselines
    # Overall: The best configuration by averaging the ranks over all inputs
    best_cfg_id_overall = (
        icm[["ranks"]].groupby("configurationID").mean().idxmin().item()
    )

    # Metric: The best configuration per performance metric
    best_cfg_id_per_metric = (
        icm_ranked_measures.groupby("configurationID").mean().idxmin()
    )

    # Common: The most common configuration in the Pareto fronts
    most_common_cfg_id = (
        dataset[["configurationID"] + [config_features.columns[0]]]
        .groupby(["configurationID"], as_index=False)
        .count()
        .sort_values(by=config_features.columns[0], ascending=False)
        .iloc[0]
        .configurationID
    )

    num_test_inputs = icm_test.index.get_level_values(0).nunique()

    overall_ranks = icm_test.query("configurationID == @best_cfg_id_overall").ranks
    assert (
        overall_ranks.shape[0] == num_test_inputs
    ), "Not all inputs are covered by the overall configurations"

    metric_ranks = (
        icm_test.query("configurationID.isin(@best_cfg_id_per_metric.values)")
        .groupby("inputname")
        .mean()
        .ranks
    )
    assert (
        metric_ranks.shape[0] == num_test_inputs
    ), "Not all inputs are covered by the metric configurations"

    common_ranks = icm_test.query("configurationID == @most_common_cfg_id").ranks
    assert (
        common_ranks.shape[0] == num_test_inputs
    ), "Not all inputs are covered by the most common configuration"

    # TODO Not sure std. dev. is correct here. We sample all random configs at once.
    max_config_id = icm.index.get_level_values(1).max()
    random_configs = np.random.randint(0, max_config_id, 10) + 1
    random_ranks = icm_test.query("configurationID.isin(@random_configs)").ranks

    if verbose:
        print(
            f"Average rank of the overall best configuration: {overall_ranks.mean():.2f}+-{overall_ranks.std():.2f}"
        )
        print(
            f"Average rank of the most common configuration: {common_ranks.mean():.2f}+-{common_ranks.std():.2f}"
        )
        print(
            f"Average rank of the best configuration for all metrics: {metric_ranks.mean():.2f}+-{metric_ranks.std():.2f}"
        )
        print(
            f"Average rank of random configuration: {random_ranks.mean():.2f}+-{random_ranks.std():.2f}"
        )

    results = {}
    results["overall"] = [overall_ranks.mean(), overall_ranks.std()]
    results["metric"] = [metric_ranks.mean(), metric_ranks.std()]
    results["common"] = [common_ranks.mean(), common_ranks.std()]
    results["random"] = [random_ranks.mean(), random_ranks.std()]

    return results


def common_labels_impurity(X, y):
    label_counts = np.bincount(y)
    unique_X_count = np.unique(X, axis=0).shape[0]
    impurity = unique_X_count - label_counts.max()
    return impurity


def common_labels_impurity_multiclass(y):
    impurity = y.shape[0] - y.sum(axis=0).max()
    return impurity


class DecisionTreeClassifierWithMultipleLabels(_MultiOutputEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
        self.random_state = random_state
        self.num_classes = None

    def fit(self, X, y):
        self.num_classes = y.shape[1]
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        if len(y) == 0:
            return None

        # Impurity
        impurity = common_labels_impurity_multiclass(y)

        # Check stopping conditions
        if depth == self.max_depth or len(y) < self.min_samples_split or impurity == 0:
            return {"type": "leaf", "class": np.argmax(y.sum(axis=0))}

        # Find the best split
        best_split = None
        best_impurity = impurity
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                left_impurity = common_labels_impurity_multiclass(left_y)
                right_impurity = common_labels_impurity_multiclass(right_y)

                # TODO This is probably not ideal for cases where X can have duplicates
                weighted_impurity = (
                    len(left_y) * left_impurity + len(right_y) * right_impurity
                ) / len(y)

                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                        "left_y": left_y,
                        "right_y": right_y,
                    }

        if best_split is None:
            return {"type": "leaf", "class": np.argmax(y.sum(axis=0))}

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(
            X[best_split["left_mask"]], best_split["left_y"], depth + 1
        )
        right_subtree = self._build_tree(
            X[best_split["right_mask"]], best_split["right_y"], depth + 1
        )

        return {
            "type": "node",
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_subtree,
            "right": right_subtree,
        }

    def predict(self, X):
        return np.array([self._predict_instance(x) for x in X])

    def _predict_instance(self, x):
        node = self.tree_
        while node["type"] != "leaf":
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["class"]

    def score(self, X, y_true):
        y_pred = self.predict(X)

        # Ratio of predictions that are true, i.e. true positives
        return y_true[np.arange(y_true.shape[0]), y_pred].mean()

    def unique_leaf_values(self):
        node = self.tree_
        to_traverse = [node]
        values = []
        while len(to_traverse) > 0:
            node = to_traverse.pop()

            if node["type"] == "leaf":
                values.append(node["class"])
            else:
                to_traverse.append(node["left"])
                to_traverse.append(node["right"])

        return len(np.unique(values))


# Similar as above but can handle pandas dataframe + categorical columns
class DecisionTreeClassifierWithMultipleLabelsPandas:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for _, x in X.iterrows()])

    def _grow_tree(self, X, y, depth=0):
        n_labels = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1:
            leaf_value = np.argmax(y.sum(axis=0))
            return {"leaf_value": leaf_value}

        feature_indices = X.columns
        best_feature, best_threshold = self._best_split(X, y, feature_indices)

        if best_feature is None:
            leaf_value = np.argmax(y.sum(axis=0))
            return {"leaf_value": leaf_value}

        if X.dtypes[best_feature] == "object":  # Categorical feature
            left_indices = X[best_feature] == best_threshold
            right_indices = X[best_feature] != best_threshold
        else:  # Numerical feature
            left_indices = X[best_feature] <= best_threshold
            right_indices = X[best_feature] > best_threshold

        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def _best_split(self, X, y, feature_names):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in feature_names:
            if X.dtypes[feature] == "object":  # Categorical feature
                thresholds = X[feature].unique()
            else:  # Numerical feature
                # thresholds = np.unique(X[feature])
                thresholds = np.percentile(X[feature], [25, 50, 75])

            for threshold in thresholds:
                if X.dtypes[feature] == "object":  # Categorical feature
                    left_indices = X[feature] == threshold
                    right_indices = X[feature] != threshold
                else:  # Numerical feature
                    left_indices = X[feature] <= threshold
                    right_indices = X[feature] > threshold

                if len(y[left_indices]) > 0 and len(y[right_indices]) > 0:
                    gain = self._information_gain(y, y[left_indices], y[right_indices])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        return common_labels_impurity_multiclass(parent) - (
            weight_left * common_labels_impurity_multiclass(left_child)
            + weight_right * common_labels_impurity_multiclass(right_child)
        )

    def _traverse_tree(self, x, node):
        if "leaf_value" in node:
            return node["leaf_value"]

        if (
            pd.api.types.is_categorical_dtype(x[node["feature"]])
            or x[node["feature"]] == "object"
        ):
            if x[node["feature"]] == node["threshold"]:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            if x[node["feature"]] <= node["threshold"]:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])

    def feature_importance(self):
        importances = np.zeros(self.n_features)

        def traverse(node, importance):
            if "feature" in node:
                importances[node["feature"]] += importance
                traverse(node["left"], importance / 2)
                traverse(node["right"], importance / 2)

        traverse(self.tree, 1.0)
        return pd.Series(importances / np.sum(importances), index=self.feature_names)

    def score(self, X, y_true):
        y_pred = self.predict(X)

        # Ratio of predictions that are true, i.e. true positives
        return y_true[np.arange(y_true.shape[0]), y_pred].mean()

    def unique_leaf_values(self):
        node = self.tree
        to_traverse = [node]
        values = []
        while len(to_traverse) > 0:
            node = to_traverse.pop()

            if "leaf_value" in node:
                values.append(node["leaf_value"])
            else:
                to_traverse.append(node["left"])
                to_traverse.append(node["right"])

        return len(np.unique(values))
