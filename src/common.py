import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
        perf_matrix["fps"] = -perf_matrix[
            "fps"
        ]  # fps is the only increasing performance measure

    # Drop inputs with constant measurements
    perf_matrix = perf_matrix[
        (
            perf_matrix[["inputname"] + performances]
            .groupby("inputname")
            .transform("std")
            > 0
        ).all(axis=1)
    ]

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
        ranks[infeasible] = ranks.max() + 1

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
