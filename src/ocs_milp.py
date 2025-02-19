# %%
import cvxpy as cp
import numpy as np
import pandas as pd
import argparse
import csv
from pathlib import Path

from common import load_data
import warnings

# There's a weird cpmpy warning that we ignore
warnings.filterwarnings("ignore")

# %%


def find_optimal_configurations(
    system, num_performances=None, optimization_target="mean"
):
    """
    Find optimal configurations for a given system and performance metrics.

    Args:
        system (str): Name of the system to analyze
        performances (list): List of performance metrics to consider. If None, uses all available metrics.
        optimize_type (str): Type of optimization - either 'mean' or 'max'

    Returns:
        list: List of dictionaries containing results for each configuration count
    """
    # Data Loading
    (
        perf_matrix_initial,
        _,
        _,
        all_performances,
        _,
        _,
    ) = load_data(system=system, data_dir="./data", input_properties_type="tabular")

    if num_performances is None:
        performances = all_performances
    else:
        performances = all_performances[:num_performances]

    print(f"Using {len(performances)} performances: {performances}")

    # Normalize performance metrics
    nmdf = (
        perf_matrix_initial[["inputname"] + performances]
        .groupby("inputname", as_index=True)
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    )
    nmdf["worst_case_performance"] = nmdf[performances].max(axis=1)

    # Prepare performance matrix
    perf_matrix = pd.merge(
        perf_matrix_initial,
        nmdf,
        suffixes=("_raw", None),
        left_index=True,
        right_index=True,
    )

    # Create configuration-input performance matrix
    cip = perf_matrix[["configurationID", "inputname", "worst_case_performance"]].pivot(
        index="configurationID", columns="inputname", values="worst_case_performance"
    )

    # Setup optimization parameters
    max_configs = cip.shape[0]
    scaling_factor = 10_000
    prev_best_cost = 0
    results = []
    prev_configs = None
    M = 1e6
    
    cip = cip.iloc[:, :100]

    # Convert to costs (higher is worse)
    cip_np = cp.Parameter(
        cip.shape, 
        nonneg=True, 
        value=np.round((cip) * scaling_factor).astype(np.int64).to_numpy()
    )
    cfg_map = {i: cip.index[i] for i in range(cip_np.shape[0])}

    # Find optimal configurations for increasing numbers of configurations
    for num_configs in range(1, max_configs + 1):
        print(f"Solving for {num_configs} configs")

        # Create binary variables for configuration selection
        x = cp.Variable(cip_np.shape[0], boolean=True, name="cfg")
        
        # Variable for the cost of each input
        item_cost = cp.Variable(cip_np.shape[1], nonneg=True, name="item_cost")

        constraints = [
            cp.sum(x) == num_configs,  # Select exactly num_configs configurations
            item_cost >= 0,
            # item_cost <= scaling_factor,  # Upper bound on costs
        ]

        # For each input
        for inp_idx in range(cip_np.shape[1]):
            # item_cost[inp_idx] should equal the maximum cost among selected configurations
            # This can be modeled as: item_cost[inp_idx] >= cost[i] for each selected config i
            constraints.append(
                item_cost[inp_idx] >= cp.multiply(cip_np[:, inp_idx], x) + M * (1 - x)
            )

        # Choose objective based on optimization target
        if optimization_target == "mean":
            objective = cp.Minimize(cp.sum(item_cost))
        else:
            objective = cp.Minimize(cp.max(item_cost))

        # Add constraint to ensure we don't get worse than previous solution
        if prev_configs is not None:
            constraints.append(cp.sum(item_cost) <= prev_best_cost)

        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        
        prob.solve(solver=cp.SCIP, verbose=True)

        print(prob.status)
        if prob.status != cp.OPTIMAL:
            print(f"No optimal solution found for {num_configs} configs")
            continue

        # Extract results
        last_configs = np.where(x.value > 0.5)[0]
        real_configs = [cfg_map[i] for i in last_configs]
        real_input_cost = np.mean((item_cost.value[last_configs] / scaling_factor))  # TODO This is wrong, indexes inputs with config ids
        prev_best_cost = prob.value

        iter_result = {
            "num_configs": num_configs,
            "selected_configs": real_configs,
            "input_cost": real_input_cost.item(),
            "wcp_mean": float(cip.loc[real_configs].min(axis=0).mean()),
            "wcp_max": float(cip.loc[real_configs].min(axis=0).max()),
            "optimization_target": optimization_target,
        }
        results.append(iter_result)
        print(iter_result)

        # Stop if we've reached optimal performance
        if real_input_cost == 0 or (
            len(results) >= 2 and real_input_cost == results[-2]["input_cost"]
        ):
            print(f"Found optimal assignment with {num_configs} configs")
            break

        prev_configs = real_configs

    return results


def save_results(results, system, performances, output_dir="results"):
    """Save results to a CSV file"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create filename based on system, performances and optimization target
    perf_str = "_".join(performances) if performances else "all"
    opt_target = results[0]["optimization_target"]
    filename = f"ocs_{system}_{perf_str}_{opt_target}_results.csv"
    filepath = Path(output_dir) / filename

    # Prepare results for CSV
    csv_results = []
    for result in results:
        row = {
            "num_configs": result["num_configs"],
            "selected_configs": str(result["selected_configs"]),
            "input_cost": result["input_cost"],
            "wcp_mean": result["wcp_mean"],
            "wcp_max": result["wcp_max"],
            "optimization_target": result["optimization_target"],
        }
        csv_results.append(row)

    # Write to CSV
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)

    print(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal configurations for a system"
    )
    parser.add_argument(
        "--system", type=str, help="Name of the system to analyze", default="poppler"
    )
    parser.add_argument(
        "--performances",
        type=int,
        help="Number of performance metrics to consider",
        default=None,
    )
    parser.add_argument(
        "--optimize_type",
        type=str,
        choices=["mean", "max"],
        help="Type of optimization to perform (mean or max)",
        default="mean",
    )

    args = parser.parse_args()

    results = find_optimal_configurations(
        args.system,
        args.performances,
        args.optimize_type,
    )

    save_results(results, args.system, args.performances)


if __name__ == "__main__":
    main()
