import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import KFold

from common import load_data
from ocs_ortools import solve_min_sum_selection


def find_optimal_configurations_cv(
    system, optimization_target="mean", num_threads=1, n_splits=5, random_state=42
):
    """
    Find optimal configurations using cross-validation for a given system and performance metrics.

    Args:
        system (str): Name of the system to analyze
        optimization_target (str): Type of optimization - either 'mean' or 'max'
        num_threads (int): Number of threads to use for solving
        n_splits (int): Number of folds for cross-validation
        random_state (int): Random seed for reproducibility

    Returns:
        list: List of dictionaries containing results for each configuration count and fold
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
    scaling_factor = 100_000

    # Create KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for num_performances in range(1, len(all_performances) + 1):
        performances = all_performances[:num_performances]

        print(f"\n{system}: Using {len(performances)} performances: {performances}")

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

        # Perform cross-validation
        fold_idx = 1
        for train_idx, test_idx in kf.split(cip.columns):
            print(f"\nProcessing fold {fold_idx}/{n_splits}")

            # Split data into train and test sets
            train_inputs = cip.columns[train_idx]
            test_inputs = cip.columns[test_idx]

            # Create training matrix
            train_matrix = (
                np.round(cip[train_inputs] * scaling_factor).astype(np.int64).to_numpy()
            )
            cfg_map = {i: cip.index[i] for i in range(train_matrix.shape[0])}

            # Initialize with best single configuration
            if optimization_target == "mean":
                seed_cfg = train_matrix.mean(axis=1).argmin()
                seed_obj_value = train_matrix.mean(axis=1).min() * train_matrix.shape[1]
            else:
                seed_cfg = train_matrix.max(axis=1).argmin()
                seed_obj_value = train_matrix.max(axis=1).min()

            indices = [k for k, v in cfg_map.items() if v == seed_cfg]
            obj_value = seed_obj_value
            last_obj_value = float("inf")

            # Find optimal configurations for increasing numbers of configurations
            for num_configs in range(1, max_configs + 1):
                print(f"\nSolving for {num_configs} configs (Fold {fold_idx})")

                # Solve using training data
                indices, obj_value = solve_min_sum_selection(
                    train_matrix,
                    num_configs,
                    optimization_target,
                    prev_solution=indices,
                    prev_obj_value=obj_value,
                    num_threads=num_threads,
                )

                if optimization_target == "mean":
                    train_cost = obj_value / train_matrix.shape[1] / scaling_factor
                else:
                    train_cost = obj_value / scaling_factor

                # Extract results and evaluate on test set
                real_configs = [cfg_map[i] for i in indices]

                # Calculate metrics on training set
                train_wcp_mean = float(
                    cip.loc[real_configs][train_inputs].min(axis=0).mean()
                )
                train_wcp_max = float(
                    cip.loc[real_configs][train_inputs].min(axis=0).max()
                )

                # Calculate metrics on test set
                test_wcp_mean = float(
                    cip.loc[real_configs][test_inputs].min(axis=0).mean()
                )
                test_wcp_max = float(
                    cip.loc[real_configs][test_inputs].min(axis=0).max()
                )

                iter_result = {
                    "system": system,
                    "num_performances": len(performances),
                    "performances": performances,
                    "num_configs": num_configs,
                    "selected_configs": real_configs,
                    "fold": fold_idx,
                    "train_cost": train_cost,
                    "train_wcp_mean": train_wcp_mean,
                    "train_wcp_max": train_wcp_max,
                    "test_wcp_mean": test_wcp_mean,
                    "test_wcp_max": test_wcp_max,
                    "optimization_target": optimization_target,
                }
                results.append(iter_result)

                # Create and display formatted table
                table = Table(
                    title=f"{system} (|P| = {len(performances)}/{len(all_performances)}): Fold {fold_idx}, Configs {num_configs}",
                    box=box.ROUNDED,
                )
                table.add_column("Metric", style="cyan")
                table.add_column("Train", style="magenta")
                table.add_column("Test", style="yellow")

                table.add_row("Number of Configs", str(num_configs), "")
                table.add_row("Input Cost", f"{train_cost:.5f}", "-")
                table.add_row(
                    "WCP Mean", f"{train_wcp_mean:.5f}", f"{test_wcp_mean:.5f}"
                )
                table.add_row("WCP Max", f"{train_wcp_max:.5f}", f"{test_wcp_max:.5f}")

                console.print(table)

                # Stop if we've reached optimal performance on training set
                target_metric = (
                    train_wcp_mean if optimization_target == "mean" else train_wcp_max
                )
                if (obj_value == last_obj_value) or (target_metric < 0.00001):
                    print(f"\nFound optimal assignment with {num_configs} configs")
                    break

                last_obj_value = obj_value

            fold_idx += 1

    return results


def save_results(results, system, output_dir="results"):
    """Save results to a CSV file"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create filename based on system and optimization target
    opt_target = results[0]["optimization_target"]
    filename = f"ocs_{system}_{opt_target}_cv.csv"
    filepath = Path(output_dir) / filename

    # Prepare results for CSV
    csv_results = []
    for result in results:
        row = {
            "system": result["system"],
            "num_performances": result["num_performances"],
            "performances": str(result["performances"]),
            "num_configs": result["num_configs"],
            "selected_configs": str(result["selected_configs"]),
            "fold": result["fold"],
            "train_cost": result["train_cost"],
            "train_wcp_mean": result["train_wcp_mean"],
            "train_wcp_max": result["train_wcp_max"],
            "test_wcp_mean": result["test_wcp_mean"],
            "test_wcp_max": result["test_wcp_max"],
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
    """Print a summary table of results"""
    console = Console()

    # Group results by num_performances and num_configs
    df = pd.DataFrame(results)
    summary = (
        df.groupby(["num_performances", "num_configs"])
        .agg(
            {
                "train_wcp_mean": ["mean", "std"],
                "train_wcp_max": ["mean", "std"],
                "test_wcp_mean": ["mean", "std"],
                "test_wcp_max": ["mean", "std"],
            }
        )
        .round(4)
    )

    table = Table(
        title=f"Cross-Validation Results Summary (Type: {results[0]['optimization_target']})",
        box=box.ROUNDED,
    )

    # Add columns
    table.add_column("#P", style="magenta", justify="right")
    table.add_column("#Cfgs", style="magenta", justify="right")
    table.add_column("Train WCP Mean", style="magenta", justify="right")
    table.add_column("Test WCP Mean", style="yellow", justify="right")
    table.add_column("Train WCP Max", style="magenta", justify="right")
    table.add_column("Test WCP Max", style="yellow", justify="right")

    prev_perf_count = None

    for idx in summary.index:
        num_p, num_c = idx
        row = summary.loc[idx]

        if prev_perf_count is not None and prev_perf_count != num_p:
            table.add_section()

        table.add_row(
            str(num_p),
            str(num_c),
            f"{row[('train_wcp_mean', 'mean')]}±{row[('train_wcp_mean', 'std')]}",
            f"{row[('test_wcp_mean', 'mean')]}±{row[('test_wcp_mean', 'std')]}",
            f"{row[('train_wcp_max', 'mean')]}±{row[('train_wcp_max', 'std')]}",
            f"{row[('test_wcp_max', 'mean')]}±{row[('test_wcp_max', 'std')]}",
        )

        prev_perf_count = num_p

    console.print("\n")
    console.print(table)
    console.print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal configurations using cross-validation"
    )
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        help="Name of the system to analyze",
        default="nodejs",
    )
    parser.add_argument(
        "-ot",
        "--optimize_type",
        type=str,
        choices=["mean", "max"],
        help="Type of optimization to perform",
        default="mean",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Number of threads to use for solving",
        default=1,
    )
    parser.add_argument(
        "-k",
        "--folds",
        type=int,
        help="Number of folds for cross-validation",
        default=4,
    )
    parser.add_argument(
        "-r",
        "--random_state",
        type=int,
        help="Random seed for reproducibility",
        default=42,
    )

    args = parser.parse_args()

    results = find_optimal_configurations_cv(
        args.system,
        args.optimize_type,
        num_threads=args.threads,
        n_splits=args.folds,
        random_state=args.random_state,
    )
    print_results(results)
    save_results(results, args.system)


if __name__ == "__main__":
    main()
