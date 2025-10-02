from scipy import stats
import pandas as pd
import json



# TODO
result_list_dict = json.load("./results/wcp_1cfg_gosdt_all.json")
baseline_df = pd.DataFrame(result_list_dict)

def make_latex_table(query_columns, baselines, base_query=None, pvalue_threshold=0.05):
    # TODO Lookup table for nice column names

    col_desc = "l" * len(query_columns) + "r" * len(baselines)
    print("\\begin{tabular}{" + col_desc + "}")
    print("\\toprule")
    print(
        " & ".join(map(lambda s: s.capitalize().replace("_", " "), query_columns))
        + " & "
        + " & ".join(map(lambda s: s.capitalize().replace("_", " "), baselines))
        + "\\\\\\midrule"
    )
    if base_query is not None:
        aggdf = baseline_df.query(base_query)
    else:
        aggdf = baseline_df
    aggdf = aggdf.groupby(query_columns).mean(numeric_only=True)
    for group_values, v in aggdf.iterrows():
        # Get just this group's results from result_list_dict
        filter_dict = dict(
            zip(
                query_columns,
                [group_values] if isinstance(group_values, str) else group_values,
            )
        )
        system_results = pd.DataFrame(
            [
                d
                for d in result_list_dict
                if all(d[k] == v for k, v in filter_dict.items())
            ]
        )

        # Check if SDC is significantly better than overall
        diff_avg_sdc = system_results["sdc_avg"] - system_results["overall_avg"]
        diff_max_sdc = system_results["sdc_max"] - system_results["overall_max"]
        non_zero_diff_avg_sdc = diff_avg_sdc[diff_avg_sdc != 0]
        non_zero_diff_max_sdc = diff_max_sdc[diff_max_sdc != 0]

        # Check if overall is significantly better than SDC
        diff_avg_overall = system_results["overall_avg"] - system_results["sdc_avg"]
        diff_max_overall = system_results["overall_max"] - system_results["sdc_max"]
        non_zero_diff_avg_overall = diff_avg_overall[diff_avg_overall != 0]
        non_zero_diff_max_overall = diff_max_overall[diff_max_overall != 0]

        # Only perform test if we have enough non-zero differences
        if len(non_zero_diff_avg_sdc) >= 5:  # minimum recommended sample size
            wilcoxon_avg_sdc = stats.wilcoxon(non_zero_diff_avg_sdc, alternative="less")
            avg_marker_sdc = "^*" if wilcoxon_avg_sdc.pvalue < pvalue_threshold else ""
        else:
            avg_marker_sdc = "^-"

        if len(non_zero_diff_max_sdc) > 5:
            wilcoxon_max_sdc = stats.wilcoxon(non_zero_diff_max_sdc, alternative="less")
            max_marker_sdc = "^*" if wilcoxon_max_sdc.pvalue < pvalue_threshold else ""
        else:
            max_marker_sdc = "^-"

        if len(non_zero_diff_avg_overall) >= 5:
            wilcoxon_avg_overall = stats.wilcoxon(
                non_zero_diff_avg_overall, alternative="less"
            )
            avg_marker_overall = (
                "^*" if wilcoxon_avg_overall.pvalue < pvalue_threshold else ""
            )
        else:
            avg_marker_overall = "^-"

        if len(non_zero_diff_max_overall) > 5:
            wilcoxon_max_overall = stats.wilcoxon(
                non_zero_diff_max_overall, alternative="less"
            )
            max_marker_overall = (
                "^*" if wilcoxon_max_overall.pvalue < pvalue_threshold else ""
            )
        else:
            max_marker_overall = "^-"

        # Format each baseline's results

        def plot_column(col):
            if col == "sdc_avg":
                marker = avg_marker_sdc
            elif col == "overall_avg":
                marker = avg_marker_overall
            elif col == "sdc_max":
                marker = max_marker_sdc
            elif col == "overall_max":
                marker = max_marker_overall
            else:
                marker = ""

            result = "{avg:.1f}{marker}".format(
                avg=100 * v[col],
                # std=100 * v[f"{b}_std"],
                marker=marker,
            )  # \\pm{std:.1f}
            if marker == "^*":
                result = "\\boldsymbol{" + result + "}"
            return f"${result}$"

        formatted_results = [plot_column(c) for c in columns]

        res = " & ".join(formatted_results)
        group_identifier = " & ".join([str(filter_dict[c]) for c in query_columns])
        print(f"{group_identifier} & {res} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


# Plot per-system results
columns = ["sdc_avg", "overall_avg", "metric_avg", "average_avg"]
make_latex_table(["system"], columns)

# %%
## Print baseline table in latex
make_latex_table(["system", "num_performances"], columns)

# %%
# Plot single-metric results
make_latex_table(
    ["system", "performances"], columns, base_query="num_performances == 1"
)

# %%

rld = pd.DataFrame(result_list_dict)

sdc_max_better = (rld["sdc_max"] < rld["overall_max"]).mean()
sdc_max_better_by = (rld["sdc_max"] - rld["overall_max"]).mean()

print(f"SDC max better: {sdc_max_better:.2%} (by {sdc_max_better_by:.2})")

sdc_avg_better = (rld["sdc_avg"] < rld["overall_avg"]).mean()
sdc_avg_better_by = (rld["sdc_avg"] - rld["overall_avg"]).mean()

print(f"SDC avg better: {sdc_avg_better:.2%} (by {sdc_avg_better_by:.2})")

aggregate_over_all_systems = (
    rld[
        [
            "sdc_avg",
            "sdc_max",
            "overall_avg",
            "overall_max",
            "average_avg",
            "average_max",
        ]
    ]
    .agg(["mean", "std"])
    .round(3)
)

print(aggregate_over_all_systems)

wilcoxon_avg = stats.wilcoxon(rld["sdc_avg"], rld["overall_avg"], alternative="less")
ttest_avg = stats.ttest_ind(rld["sdc_avg"], rld["overall_avg"])
wilcoxon_max = stats.wilcoxon(rld["sdc_max"], rld["overall_max"], alternative="less")
ttest_max = stats.ttest_ind(rld["sdc_max"], rld["overall_max"])

print(f"Wilcoxon test (Avg. WCP SDC/Overall): {wilcoxon_avg.pvalue:.1e}")
print(f"T-test (Avg. WCP SDC/Overall): {ttest_avg.pvalue:.1e}")

print(f"Wilcoxon test (Max. WCP SDC/Overall): {wilcoxon_max.pvalue:.1e}")
print(f"T-test (Max. WCP SDC/Overall): {ttest_max.pvalue:.1e}")