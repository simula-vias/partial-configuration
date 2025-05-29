import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from common import load_data, NpEncoder
from create_splits import read_splits
from ocs_ortools import solve_min_sum_selection, try_load_results


def get_result(results, performances, num_configs, fold_idx):
    for r in results:
        if (
            r["num_configs"] == num_configs
            and r["fold"] == fold_idx
            and r["performances"] == performances
        ):
            return r
    return None


def find_optimal_configurations_cv(
    system, optimization_target="mean", num_threads=1, resume_results=None
):
    """
    Find optimal configurations using cross-validation for a given system and performance metrics.
    Uses pre-defined splits from data/splits.json.

    Args:
        system (str): Name of the system to analyze
        optimization_target (str): Type of optimization - either 'mean' or 'max'
        num_threads (int): Number of threads to use for solving

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

    results = [] if resume_results is None else resume_results.copy()
    console = Console()
    scaling_factor = 10_000

    # Build a set of completed (num_performances, num_configs, optimization_target) for fast skip
    completed = set()
    if results:
        for r in results:
            completed.add((tuple(r["performances"]), r["fold"], r["num_configs"]))

    # Load pre-defined splits from splits.json
    try:
        splits = read_splits(system, filepath="./data/splits.json")
        n_splits = len(splits)
        print(f"Loaded {n_splits} pre-defined splits for {system} from splits.json")
    except Exception as e:
        print(f"Error loading splits: {str(e)}")
        return []

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

        # Perform cross-validation using pre-defined splits
        fold_idx = 1
        for train_inputs_list, test_inputs_list in splits:
            print(f"\nProcessing fold {fold_idx}/{n_splits}")

            # Filter inputs that exist in the dataset
            train_inputs = [inp for inp in train_inputs_list if inp in cip.columns]
            test_inputs = [inp for inp in test_inputs_list if inp in cip.columns]

            print(f"Train inputs: {len(train_inputs)}, Test inputs: {len(test_inputs)}")

            # Create training matrix
            train_matrix = (
                np.round(cip[train_inputs] * scaling_factor).astype(np.int64).to_numpy()
            )
            cfg_map = {i: cip.index[i] for i in range(train_matrix.shape[0])}

            # Initialize with best single configuration
            if optimization_target == "mean":
                seed_cfg = train_matrix.mean(axis=1).argmin()
                seed_obj_value = train_matrix.mean(axis=1).min() * train_matrix.shape[1]
            elif optimization_target == "max":
                seed_cfg = train_matrix.max(axis=1).argmin()
                seed_obj_value = train_matrix.max(axis=1).min()
            else:
                # For "both" optimization target, start with the max optimization
                seed_cfg = train_matrix.max(axis=1).argmin()
                max_mean_value = scaling_factor * train_matrix.shape[0]
                max_obj_scale_factor = scaling_factor
                while max_obj_scale_factor < max_mean_value:
                    max_obj_scale_factor *= 10

                max_value = train_matrix.max(axis=1).min()
                mean_value = train_matrix.mean(axis=1).min() * train_matrix.shape[1]
                seed_obj_value = max_obj_scale_factor * max_value + mean_value
                seed_obj_value = None

            indices = [k for k, v in cfg_map.items() if v == seed_cfg]
            obj_value = seed_obj_value
            last_obj_value = float("inf")

            # Find optimal configurations for increasing numbers of configurations
            for num_configs in range(1, max_configs + 1):
                perf_key = (tuple(performances), fold_idx, num_configs)
                # Skip if already done
                if perf_key in completed:
                    continue

                # Check if previous iteration result triggers early stopping
                if (
                    resume_results is not None
                    and optimization_target != "max"
                    and num_configs >= 2
                    and len(results) >= 2
                ):
                    last_result = get_result(
                        results, performances, num_configs - 1, fold_idx
                    )
                    sec_last_result = get_result(
                        results, performances, num_configs - 2, fold_idx
                    )
                    if last_result["train_cost"] == sec_last_result["train_cost"]:
                        break

                print(f"\nSolving for {num_configs} configs (Fold {fold_idx})")

                # Solve using training data
                all_solutions, obj_value = solve_min_sum_selection(
                    train_matrix,
                    num_configs,
                    optimization_target,
                    warm_start_solution=indices,
                    prev_obj_value=obj_value,
                    num_threads=num_threads,
                    scaling_factor=scaling_factor,
                )

                train_cost = obj_value

                for indices in all_solutions:
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

                    # Determine which configuration is best for each input
                    # For each input, find the configuration with the minimum performance value
                    selected_configs_df = cip.loc[real_configs]
                    input_to_config_map = {}

                    for input_name in selected_configs_df.columns:
                        if input_name in train_inputs:
                            # Find the configuration with the minimum performance value for this input
                            best_config = selected_configs_df[input_name].idxmin()
                            input_to_config_map[input_name] = best_config

                    iter_result = {
                        "system": system,
                        "num_performances": len(performances),
                        "performances": performances,
                        "num_configs": num_configs,
                        "selected_configs": real_configs,
                        "fold": fold_idx,
                        "train_inputs": train_inputs,
                        "test_inputs": test_inputs,
                        "train_cost": train_cost,
                        "train_wcp_mean": train_wcp_mean,
                        "train_wcp_max": train_wcp_max,
                        "test_wcp_mean": test_wcp_mean,
                        "test_wcp_max": test_wcp_max,
                        "optimization_target": optimization_target,
                        "input_to_config_map": input_to_config_map,
                    }
                    results.append(iter_result)

                completed.add(perf_key)
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
                table.add_row("Number of Solutions", str(len(all_solutions)))

                console.print(table)

                # This is not a reliable stopping criterion for max
                # For wcp_max, we can have iterations without improvement,
                # because we must first cover all worst-case items through more configs
                if (
                    optimization_target != "max"
                    and num_configs >= 2
                    and len(results) >= 2
                    and obj_value == last_obj_value
                ):
                    print(f"\nNo improvement after {num_configs} configs")
                    break

                last_obj_value = obj_value

            fold_idx += 1

    return results


def save_results(results, system, output_dir="results"):
    """Save results to JSON file"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create filenames based on system and optimization target
    opt_target = results[0]["optimization_target"]
    json_filename = f"ocs_{system}_{opt_target}_cv.json"
    json_filepath = Path(output_dir) / json_filename

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

    print(f"Results saved to {json_filepath}")


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
        description="Find optimal configurations using cross-validation with pre-defined splits"
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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start fresh and ignore any existing results file",
    )
    args = parser.parse_args()

    resume_results = (
        None
        if args.reset
        else try_load_results(args.system, args.optimize_type, suffix="_cv")
    )
    results = find_optimal_configurations_cv(
        args.system,
        args.optimize_type,
        num_threads=args.threads,
        resume_results=resume_results,
    )
    print_results(results)
    save_results(results, args.system)


if __name__ == "__main__":
    main()
