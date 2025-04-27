import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from rich import box
from rich.console import Console
from rich.table import Table

from common import load_data, NpEncoder


def solve_min_sum_selection(
    matrix,
    k,
    optimization_target="mean",
    prev_solution=None,
    prev_obj_value=None,
    num_threads=1,
    scaling_factor=10_000,
):
    """
    Solves the row selection problem where:
    - Given NxM matrix
    - Select k rows
    - Minimize sum of column minimums among selected rows

    Args:
        matrix: List of lists representing NxM matrix
        k: Number of rows to select
        optimization_target: Either "mean" or "max"
        prev_solution: Optional list of previously selected indices for warm start
        prev_obj_value: Optional previous objective value for warm start
        num_threads: Number of threads to use for solving (default: 1)

    Returns:
        selected_rows: List of indices of selected rows
        objective_value: Sum of column minimums
    """
    N = len(matrix)  # Number of rows
    M = len(matrix[0])  # Number of columns

    # Create the solver
    solver = pywraplp.Solver.CreateSolver("CP_SAT")

    # Decision Variables
    # x[i] = 1 if row i is selected, 0 otherwise
    x = [solver.BoolVar(f"x_{i}") for i in range(N)]

    # y[j] represents the minimum value in column j among selected rows
    y = [solver.IntVar(0, int(matrix[:, j].max()), f"y_{j}") for j in range(M)]

    # min_indicator[i,j] = 1 if row i provides the minimum for column j
    min_indicator = {}
    for i in range(N):
        for j in range(M):
            min_indicator[i, j] = solver.BoolVar(f"min_indicator_{i}_{j}")

    def get_mean_objective(y):
        return solver.Sum(y)

    def get_max_objective(y):
        max_val = int(max(max(row) for row in matrix))
        objective = solver.IntVar(0, max_val, "max_val")
        for j in range(M):
            solver.Add(objective >= y[j])
        return objective

    if optimization_target == "mean":
        objective = get_mean_objective(y)
    elif optimization_target == "max":
        objective = get_max_objective(y)
    else:
        # TODO Check two phase optimization, first max then mean
        # More complex for the solver, but better for our experiments

        max_mean_value = scaling_factor * N
        max_obj_scale_factor = scaling_factor
        while max_obj_scale_factor < max_mean_value:
            max_obj_scale_factor *= 10

        mean_objective = get_mean_objective(y)
        max_objective = get_max_objective(y)
        objective = max_obj_scale_factor * max_objective + mean_objective

    solver.Minimize(objective)

    # Constraints
    big_M = max(max(row) for row in matrix) + 1

    # 1. Select exactly k rows
    solver.Add(solver.Sum(x) == k)

    # 2. Column minimum constraints
    for j in range(M):
        for i in range(N):
            # The minimum can only be selected if the row is selected
            solver.Add(min_indicator[i, j] <= x[i])

            # The minimum value must be at least as large as the selected value
            solver.Add(y[j] >= matrix[i][j] - (1 - min_indicator[i, j]) * big_M)
            solver.Add(y[j] <= matrix[i][j] + (1 - min_indicator[i, j]) * big_M)

            solver.Add(y[j] <= matrix[i][j] + (1 - x[i]) * big_M)

        # Ensure exactly one minimum value is selected per column
        solver.Add(solver.Sum(min_indicator[i, j] for i in range(N)) == 1)
        solver.Add(y[j] <= solver.Sum([matrix[i][j] * x[i] for i in range(N)]))

    # Warm start if previous solution provided
    if prev_solution is not None:
        solver.SetHint(x, [1 if i in prev_solution else 0 for i in range(N)])

    if prev_obj_value is not None:
        solver.Add(objective <= prev_obj_value)

    solver.SetNumThreads(num_threads)

    # solver.EnableOutput()

    # Solve the problem
    status = solver.Solve()

    # Check if optimal solution found
    if status != pywraplp.Solver.OPTIMAL:
        print(f"Status: {status}")
        print(f"Objective value: {solver.Objective().Value()}")
        raise ValueError("Not optimal")

    assert solver.VerifySolution(1e-7, True)

    # Extract results
    selected_rows = [int(i) for i in range(N) if x[i].solution_value() > 0.5]
    objective_value = solver.Objective().Value()

    print(f"Objective value: {objective_value}")

    return selected_rows, objective_value


def find_optimal_configurations(system, optimization_target="mean", num_threads=1):
    """
    Find optimal configurations for a given system and performance metrics.

    Args:
        system (str): Name of the system to analyze
        optimization_target (str): Type of optimization - either 'mean' or 'max'
        num_threads (int): Number of threads to use for solving (default: 1)

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

    for num_performances in range(1, len(all_performances) + 1):
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
            seed_cfg = cip_np.mean(axis=1).argmin()
            seed_obj_value = cip_np.mean(axis=1).min() * cip_np.shape[1]
        elif optimization_target == "max":
            seed_cfg = cip_np.max(axis=1).argmin()
            seed_obj_value = cip_np.max(axis=1).min()
        else:
            seed_cfg = None
            seed_obj_value = None

        indices = [k for k, v in cfg_map.items() if v == seed_cfg]
        obj_value = seed_obj_value

        # Find optimal configurations for increasing numbers of configurations
        for num_configs in range(1, max_configs + 1):
            print(f"\nSolving for {num_configs} configs")

            indices, obj_value = solve_min_sum_selection(
                cip_np,
                num_configs,
                optimization_target,
                prev_solution=indices,
                prev_obj_value=obj_value,
                num_threads=num_threads,
            )

            input_cost = obj_value

            # Extract results
            real_configs = [cfg_map[i] for i in indices]
            wcp_mean = float(cip.loc[real_configs].min(axis=0).mean())
            wcp_max = float(cip.loc[real_configs].min(axis=0).max())

            # Determine which configuration is best for each input
            # For each input, find the configuration with the minimum performance value
            selected_configs_df = cip.loc[real_configs]
            input_to_config_map = {}

            for input_name in selected_configs_df.columns:
                # Find the configuration with the minimum performance value for this input
                best_config = selected_configs_df[input_name].idxmin()
                input_to_config_map[input_name] = best_config

            iter_result = {
                "system": system,
                "num_performances": len(performances),
                "performances": performances,
                "num_configs": num_configs,
                "selected_configs": real_configs,
                "input_cost": input_cost,
                "wcp_mean": wcp_mean,
                "wcp_max": wcp_max,
                "optimization_target": optimization_target,
                "input_to_config_map": input_to_config_map,
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
            table.add_row("Input Cost", f"{input_cost:.5f}")
            table.add_row("WCP Mean", f"{wcp_mean:.5f}")
            table.add_row("WCP Max", f"{wcp_max:.5f}")
            table.add_row("Optimization Target", optimization_target)

            console.print(table)

            # TODO The lower bound is obviously to pick all configurations?!?!?
            # And this we can calculate beforehand from the dataset.
            # Am I stupid?
            # At the same time, OR-Tools figured that out quite qickly during search anyway

            # This is not a reliable stopping criterion for max
            # For wcp_max, we can have iterations without improvement,
            # because we must first cover all worst-case items through more configs
            if (
                optimization_target != "max"
                and num_configs >= 2
                and len(results) >= 2
                and input_cost == results[-2]["input_cost"]
            ):
                print(f"\nNo improvement after {num_configs} configs")
                break

    return results


def save_results(results, system, output_dir="results"):
    """Save results to CSV and JSON files"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create filenames based on system and optimization target
    opt_target = results[0]["optimization_target"]
    csv_filename = f"ocs_{system}_{opt_target}.csv"
    json_filename = f"ocs_{system}_{opt_target}.json"
    csv_filepath = Path(output_dir) / csv_filename
    json_filepath = Path(output_dir) / json_filename

    # Prepare results for CSV
    csv_results = []
    for result in results:
        row = {
            "system": result["system"],
            "num_performances": result["num_performances"],
            "performances": str(result["performances"]),
            "num_configs": result["num_configs"],
            "selected_configs": str(result["selected_configs"]),
            "input_cost": result["input_cost"],
            "wcp_mean": result["wcp_mean"],
            "wcp_max": result["wcp_max"],
            "optimization_target": result["optimization_target"],
            "input_to_config_map": str(result["input_to_config_map"]),
        }
        csv_results.append(row)

    # Write to CSV
    with open(csv_filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)

    # Prepare results for JSON
    # We need to convert some non-serializable types to serializable ones
    json_results = []
    for result in results:
        # Create a copy of the result to avoid modifying the original
        json_result = result.copy()

        # Convert numpy arrays and other non-serializable types to lists
        if isinstance(json_result["performances"], np.ndarray):
            json_result["performances"] = json_result["performances"].tolist()

        # Convert selected_configs to list if it's not already
        if not isinstance(json_result["selected_configs"], list):
            json_result["selected_configs"] = list(json_result["selected_configs"])

        # Ensure all keys in input_to_config_map are strings for JSON compatibility
        input_to_config_map = {}
        for k, v in json_result["input_to_config_map"].items():
            input_to_config_map[str(k)] = v
        json_result["input_to_config_map"] = input_to_config_map

        json_results.append(json_result)

    # Write to JSON
    with open(json_filepath, "w") as f:
        json.dump(json_results, f, indent=2, cls=NpEncoder)

    print(f"Results saved to {csv_filepath} and {json_filepath}")


def print_results(results):
    """Print a summary table of results"""
    console = Console()

    table = Table(
        title=f"Optimization Results Summary (Type: {results[0]['optimization_target']})",
        box=box.ROUNDED,
    )

    # Add columns
    table.add_column("#P", style="magenta", justify="right")
    table.add_column("#Cfgs", style="magenta", justify="right")
    table.add_column("Input Cost", style="magenta", justify="right")
    table.add_column("WCP Max", style="magenta", justify="right")
    table.add_column("WCP Mean", style="magenta", justify="right")

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
            f"{result['wcp_max']:.4f}",
            f"{result['wcp_mean']:.4f}",
        )

        prev_perf_count = result["num_performances"]

    console.print("\n")
    console.print(table)
    console.print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal configurations for a system using OR-Tools"
    )
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        help="Name of the system to analyze",
        default="gcc",
    )
    parser.add_argument(
        "-ot",
        "--optimize_type",
        type=str,
        choices=["mean", "max", "both"],
        help="Type of optimization to perform",
        default="both",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Number of threads to use for solving",
        default=1,
    )

    args = parser.parse_args()

    results = find_optimal_configurations(
        args.system,
        args.optimize_type,
        num_threads=args.threads,
    )
    print_results(results)
    save_results(results, args.system)


if __name__ == "__main__":
    main()
