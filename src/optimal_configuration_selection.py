# %%
from cpmpy import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import load_data

# %%
# Data Loading
random_state = 1234
test_size = 0.40
rank_by_domination_count = True
system = "x264"

performances = []
# performances = ["fps", "cpu"]
# performances = ["kbs", "fps"]
# performances = ["kbs", "etime"]

(
    perf_matrix_initial,
    input_features,
    config_features,
    all_performances,
    input_preprocessor,
    config_preprocessor,
) = load_data(system=system, data_dir="../data", input_properties_type="tabular")

if len(performances) == 0:
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
    lambda x: x["worst_case_performance"].argsort() + 1
)
# We adjust the WCP by expressing it as the difference from the best WCP, i.e. the best WCP is always 0
perf_matrix["worst_case_performance"] = (
    perf_matrix[["inputname", "worst_case_performance"]]
    .groupby("inputname", as_index=True)
    .transform(lambda x: x - x.min())
)

all_perf_raw = [f"{p}_raw" for p in performances]
all_perf_norm = [f"{p}" for p in performances]


# Split data
train_inp, test_inp = train_test_split(
    perf_matrix["inputname"].unique(),
    test_size=test_size,
    random_state=random_state,
)

# This is only to make our evaluation simpler
train_inp = sorted(train_inp)
test_inp = sorted(test_inp)

# Prepare baseline evaluation
icm = (
    perf_matrix[perf_matrix.inputname.isin(train_inp)][
        ["inputname", "configurationID", "worst_case_performance"]
    ]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)
icm_ranked_measures = icm.groupby(
    "inputname"
).transform(  # Go from measured values to ranks within each input group
    lambda x: x.argsort() + 1
)
icm["ranks"] = icm.groupby("inputname", group_keys=False).apply(
    lambda x: x["worst_case_performance"].argsort() + 1
)


icm_test = (
    perf_matrix[~perf_matrix.inputname.isin(train_inp)][
        ["inputname", "configurationID", "worst_case_performance"]
    ]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)
icm_test["ranks"] = icm_test.groupby("inputname", group_keys=False).apply(
    lambda x: x["worst_case_performance"].argsort() + 1
)

# Full dataset of input features + config features that are in the first rank
dataset_baselines = icm.join(config_features).join(input_features).reset_index()


def eval_prediction(pred_cfg_test):
    inp_pred_map = pd.DataFrame(
        zip(test_inp, pred_cfg_test), columns=["inputname", "configurationID"]
    )
    return icm_test.merge(inp_pred_map, on=["inputname", "configurationID"])[
        "worst_case_performance"
    ].mean()


# %%
cip = perf_matrix[["configurationID", "inputname", "worst_case_performance"]].pivot(
    index="configurationID", columns="inputname", values="worst_case_performance"
)
# Find all rows in cip where there is another row where all values are lower, i.e. it's dominated
dominated = np.array(
    [np.any([np.all(row2 < row1) for row2 in cip.values]) for row1 in cip.values]
)

# A dominated configuration can never be relevant, so we can remove it
non_dominated_configs = cip.index[~dominated]
print(
    f"Number of non-dominated configurations: {len(non_dominated_configs)}/{len(cip)}"
)
# cip["wcp_mean"] = cip.mean(axis=1)
# cip["wcp_max"] = cip.max(axis=1)

# %%
import warnings
# There's a weird cpmpy warning that we ignore
warnings.filterwarnings("ignore")

max_configs = cip.shape[0]

scaling_factor = 10_000

best_wcp = 0
results = []

for num_configs in range(1, max_configs + 1):
    print(f"Solving for {num_configs} configs")
   
    cip_np = np.round((1 - cip) * scaling_factor).astype(np.int64).to_numpy()
    input_cost_ub = scaling_factor

    # Create boolean variables for each configuration
    x = boolvar(shape=cip_np.shape[0], name="cfg")

    # One int per input to store the minimum WCP for that input
    item_cost = intvar(shape=cip_np.shape[1], lb=0, ub=input_cost_ub)

    # Ensure all values used in calculations are integers
    for inp_idx in range(cip_np.shape[1]):
        effective_cost = cip_np[:, inp_idx] * x
        item_cost[inp_idx] = max(effective_cost)

    obj = sum(item_cost)
    m = Model(sum(x) == num_configs, maximize=obj)

    if best_wcp > 0:
        m += obj >= best_wcp

    # Solve the model
    solve_result = m.solve()
    print("Value:", solve_result)

    real_configs = np.where(x.value())[0] + 1
    real_input_cost = np.mean(1 - (item_cost.value() / scaling_factor))
    best_wcp = max(best_wcp, obj.value())  # Update best_wcp with current solution

    iter_result = {
        "num_configs": num_configs,
        "selected_configs": real_configs,
        "input_cost": real_input_cost,
        "wcp_mean": cip.loc[real_configs].min(axis=0).mean(),
        "wcp_max": cip.loc[real_configs].min(axis=0).max(),
    }
    results.append(iter_result)
    print(iter_result)

    if real_input_cost == 0:
        print(f"Found optimal assignment with {num_configs} configs")
        break




# %%
import itertools

best_wcp = 0
best_configs = None

for selected_configs in itertools.combinations(cip.index - 1, num_configs):
    wcp = cip_np[selected_configs, :].max(axis=0).mean()

    print(selected_configs, wcp)

    if wcp > best_wcp:
        best_wcp = wcp
        best_configs = selected_configs

print(f"Best WCP: {best_wcp}")
print(f"Best configs: {best_configs}")
print(f"WCP: {cip.loc[list(best_configs)].min(axis=0).mean()}")
