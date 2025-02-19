# %%
# import cvxpy as cp
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pulp as pl
from rich import box
from rich.console import Console
from rich.table import Table

from common import load_data


def solve_min_sum_selection(matrix, k, optimization_target="mean", prev_solution=None):
    """
    Solves the row selection problem where:
    - Given NxM matrix
    - Select k rows
    - Minimize sum of column minimums among selected rows

    Args:
        matrix: List of lists representing NxM matrix
        k: Number of rows to select

    Returns:
        selected_rows: List of indices of selected rows
        objective_value: Sum of column minimums
    """
    # N, M = matrix.shape
    N = len(matrix)  # Number of rows
    M = len(matrix[0])  # Number of columns

    # Create the model
    prob = pl.LpProblem("MinSumSelection", pl.LpMinimize)

    # Decision Variables
    # x[i] = 1 if row i is selected, 0 otherwise
    x = pl.LpVariable.dicts("row", range(N), cat="Binary")

    # y[j] represents the minimum value in column j among selected rows
    y = pl.LpVariable.dicts("col_min", range(M))

    min_indicator = pl.LpVariable.dicts(
        "min_indicator", ((i, j) for i in range(N) for j in range(M)), cat="Binary"
    )

    # Objective: Minimize sum of column minimums
    if optimization_target == "mean":
        prob += pl.lpSum(y[j] for j in range(M))
    else:
        max_val = pl.LpVariable("max_val", lowBound=0)
        for j in range(M):
            prob += max_val >= y[j]
        prob += max_val

    # Constraints
    big_M = max(max(row) for row in matrix) + 1

    # 1. Select exactly k rows
    prob += pl.lpSum(x[i] for i in range(N)) == k

    for j in range(M):
        for i in range(N):
            # The minimum can only be selected if the row is selected
            prob += min_indicator[i, j] <= x[i]

            # The minimum value must be at least as large as the selected value
            prob += y[j] >= matrix[i][j] - (1 - min_indicator[i, j]) * big_M
            prob += y[j] <= matrix[i][j] + (1 - min_indicator[i, j]) * big_M

            prob += y[j] <= matrix[i][j] + (1 - x[i]) * big_M

        # Ensure exactly one minimum value is selected per column
        prob += pl.lpSum(min_indicator[i, j] for i in range(N)) == 1

        prob += y[j] <= pl.lpSum(matrix[i][j] * x[i] for i in range(N))
        prob += y[j] >= 0

    if prev_solution is not None:
        for i in range(N):
            x[i].setInitialValue(1 if i in prev_solution else 0)

    # Solve the problem
    solver = pl.getSolver(
        "PULP_CBC_CMD", threads=8, msg=1, warmStart=prev_solution is not None
    )
    status = prob.solve(solver)

    # Print debug information
    print(f"Status: {pl.LpStatus[status]}")

    if status != pl.LpStatusOptimal:
        print(f"Status: {pl.LpStatus[status]}")
        print(f"Objective value: {pl.value(prob.objective)}")
        raise ValueError("Not optimal")

    # Extract results
    selected_rows = [i for i in range(N) if pl.value(x[i]) > 0.5]
    objective_value = pl.value(prob.objective)

    if optimization_target == "mean":
        objective_value = objective_value / M

    print(f"Objective value: {objective_value}")

    return selected_rows, objective_value


# %%


def find_optimal_configurations(system, optimization_target="mean"):
    """
    Find optimal configurations for a given system and performance metrics.

    Args:
        system (str): Name of the system to analyze
        optimization_target (str): Type of optimization - either 'mean' or 'max'

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

    results = []
    console = Console()
    scaling_factor = 10_000

    for num_performances in range(1, 2):  # len(all_performances) + 1):
        performances = all_performances[:num_performances]

        print(f"{system}: Using {len(performances)} performances: {performances}")

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
        cip = perf_matrix[
            ["configurationID", "inputname", "worst_case_performance"]
        ].pivot(
            index="configurationID",
            columns="inputname",
            values="worst_case_performance",
        )

        # Setup optimization parameters
        max_configs = cip.shape[0]

        cip_np = np.round(cip * scaling_factor).astype(np.int64).to_numpy()
        cfg_map = {i: cip.index[i] for i in range(cip_np.shape[0])}

        if optimization_target == "mean":
            seed_cfg = cip.mean(axis=1).idxmin()
        else:
            seed_cfg = cip.max(axis=1).idxmin()

        indices = [k for k, v in cfg_map.items() if v == seed_cfg]

        # Find optimal configurations for increasing numbers of configurations
        for num_configs in range(1, max_configs + 1):
            print(f"\nSolving for {num_configs} configs")

            indices, obj_value = solve_min_sum_selection(
                cip_np, num_configs, optimization_target, prev_solution=indices
            )

            # Extract results
            real_configs = [cfg_map[i] for i in indices]
            wcp_mean = float(cip.loc[real_configs].min(axis=0).mean())
            wcp_max = float(cip.loc[real_configs].min(axis=0).max())

            target_result = wcp_mean if optimization_target == "mean" else wcp_max

            iter_result = {
                "system": system,
                "num_performances": len(performances),
                "performances": performances,
                "num_configs": num_configs,
                "selected_configs": real_configs,
                "input_cost": obj_value / scaling_factor,
                "wcp_mean": wcp_mean,
                "wcp_max": wcp_max,
                "optimization_target": optimization_target,
            }
            results.append(iter_result)

            # Create and display formatted table
            table = Table(
                title=f"{system} (|P| = {len(performances)}/{len(all_performances)}): Iteration {num_configs}",
                box=box.ROUNDED,
            )
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Number of Configs", str(num_configs))
            table.add_row("Selected Configs", str(real_configs))
            table.add_row("Input Cost", f"{obj_value / scaling_factor:.5f}")
            table.add_row("WCP Mean", f"{wcp_mean:.5f}")
            table.add_row("WCP Max", f"{wcp_max:.5f}")
            table.add_row("Optimization Target", optimization_target)

            console.print(table)

            # Stop if we've reached optimal performance
            optimization_target_column_name = (
                "wcp_mean" if optimization_target == "mean" else "wcp_max"
            )
            if target_result < 0.00005 or (
                num_configs >= 2
                and target_result == results[-2][optimization_target_column_name]
            ):
                print(f"\nFound optimal assignment with {num_configs} configs")
                break

    return results


def save_results(results, system, output_dir="results"):
    """Save results to a CSV file"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create filename based on system and optimization target
    opt_target = results[0]["optimization_target"]
    filename = f"ocs_{system}_{opt_target}.csv"
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


def print_results(results):
    """
    Print a summary table of results with mean and max optimization results

    Args:
        results_mean: List of result dictionaries from mean optimization
        results_max: Optional list of result dictionaries from max optimization
    """
    console = Console()

    table = Table(
        title=f"Optimization Results Summary (Type: {results[0]['optimization_target']})",
        box=box.ROUNDED,
    )

    # Add columns
    table.add_column("#P", style="magenta", justify="right")
    table.add_column("#Cfgs", style="magenta", justify="right")
    table.add_column("Input Cost", style="magenta", justify="right")
    table.add_column("WCP Mean", style="magenta", justify="right")
    table.add_column("WCP Max", style="magenta", justify="right")

    # Track previous performance count to add dividers
    prev_perf_count = None

    # Add results
    for result in results:
        # Add divider if performance count changes
        if (
            prev_perf_count is not None
            and prev_perf_count != result["num_performances"]
        ):
            table.add_section()

        table.add_row(
            str(result["num_performances"]),
            str(result["num_configs"]),
            f"{result['input_cost']:.4f}",
            f"{result['wcp_mean']:.4f}",
            f"{result['wcp_max']:.4f}",
        )

        prev_perf_count = result["num_performances"]

    console.print("\n")
    console.print(table)
    console.print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal configurations for a system"
    )
    parser.add_argument(
        "--system",
        type=str,
        help="Name of the system to analyze",
        default="imagemagick",
    )
    parser.add_argument(
        "--optimize_type",
        type=str,
        choices=["mean", "max"],
        help="Type of optimization to perform (mean, max, or both)",
        default="mean",
    )

    args = parser.parse_args()

    results = find_optimal_configurations(
        args.system,
        args.optimize_type,
    )
    print_results(results)
    save_results(results, args.system)


if __name__ == "__main__":
    main()
