from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .simulation import (
    DEFAULT_POLICIES,
    ONLINE_POLICIES,
    build_profile_pool_from_patient_table,
    default_scenarios,
    experiment_1_oracle_gap,
    experiment_2_uncertainty,
    experiment_3_prediction_quality,
    run_experiment_grid,
    save_core_figures,
    smoke_test_scenarios,
    summarize_results,
)


def load_profile_pool(args: argparse.Namespace):
    if not args.patient_csv:
        return None
    table = pd.read_csv(args.patient_csv)
    return build_profile_pool_from_patient_table(
        table,
        triss_col=args.triss_col,
        patient_type_col=args.patient_type_col,
        sample_limit=args.patient_sample_limit,
    )


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EMS online vs offline transport optimization simulation."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="Smoke is fast; full uses the research-sized scenario suite.",
    )
    parser.add_argument("--seeds", type=int, default=2, help="Number of seeds to evaluate.")
    parser.add_argument(
        "--experiment",
        choices=("all", "grid", "oracle-gap", "planning-gain", "prediction-quality"),
        default="all",
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=None,
        help="Policies for --experiment grid. Defaults to nearest/fastest/survival/lookahead.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV and figures.")
    parser.add_argument("--figures", action="store_true", help="Save basic matplotlib figures.")
    parser.add_argument("--patient-csv", default=None, help="Optional patient table with TRISS column.")
    parser.add_argument("--triss-col", default="TRISS")
    parser.add_argument("--patient-type-col", default=None)
    parser.add_argument("--patient-sample-limit", type=int, default=None)
    args = parser.parse_args()

    scenarios = smoke_test_scenarios() if args.mode == "smoke" else default_scenarios()
    seeds = range(max(1, args.seeds))
    out_dir = Path(args.output_dir)
    profile_pool = load_profile_pool(args)

    if args.experiment in {"all", "grid"}:
        policies = tuple(args.policies) if args.policies else DEFAULT_POLICIES
        results = run_experiment_grid(scenarios, seeds, policies=policies, profile_pool=profile_pool)
        summary = summarize_results(results)
        write_table(results, out_dir / "grid_results.csv")
        write_table(summary, out_dir / "grid_summary.csv")
        print("\nGrid summary")
        print(summary.round(3).to_string(index=False))
        if args.figures:
            save_core_figures(results, out_dir / "figures")

    if args.experiment in {"all", "oracle-gap"}:
        results, comparison = experiment_1_oracle_gap(scenarios, seeds, profile_pool=profile_pool)
        write_table(results, out_dir / "experiment_1_oracle_gap_results.csv")
        write_table(comparison, out_dir / "experiment_1_oracle_gap_summary.csv")
        print("\nExperiment 1: Offline oracle vs online policies")
        print(comparison.round(3).to_string(index=False))

    if args.experiment in {"all", "planning-gain"}:
        results, by_scenario, by_regime = experiment_2_uncertainty(
            scenarios, seeds, profile_pool=profile_pool
        )
        write_table(results, out_dir / "experiment_2_planning_gain_results.csv")
        write_table(by_scenario, out_dir / "experiment_2_planning_gain_by_scenario.csv")
        write_table(by_regime, out_dir / "experiment_2_planning_gain_by_regime.csv")
        print("\nExperiment 2: Where planning matters")
        print(by_scenario.round(3).to_string(index=False))
        print()
        print(by_regime.round(3).to_string(index=False))
        if args.figures:
            save_core_figures(
                run_experiment_grid(
                    scenarios,
                    seeds,
                    policies=("offline_oracle", "online_greedy", "online_mc_perfect"),
                    profile_pool=profile_pool,
                ),
                out_dir / "figures",
            )

    if args.experiment in {"all", "prediction-quality"}:
        results, comparison = experiment_3_prediction_quality(scenarios, seeds, profile_pool=profile_pool)
        write_table(results, out_dir / "experiment_3_prediction_quality_results.csv")
        write_table(comparison, out_dir / "experiment_3_prediction_quality_summary.csv")
        print("\nExperiment 3: Prediction quality robustness")
        print(comparison.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
