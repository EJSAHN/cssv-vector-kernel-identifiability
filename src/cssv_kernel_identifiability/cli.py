from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import parse_scales, run_empirical_analysis, run_simulation_study
from .export import write_tables, write_excel_workbook


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze CSSV protective-ring data and kernel identifiability")
    parser.add_argument("--project-root", default=".", help="Repository root. Default: current directory")
    parser.add_argument("--output-dir", default="results", help="Output directory relative to project root unless absolute")
    parser.add_argument("--denominator-mode", choices=["reported", "reconstructed", "table", "expected_geometry"], default="reported")
    parser.add_argument("--scales", default="2.4,3.6,4.8,7.2,9.6,14.4,19.2,28.8")
    parser.add_argument("--true-scales", default="4.8,9.6,28.8")
    parser.add_argument("--n-sim", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--year-effects", default=None,
                        help="CSV of fixed year effects (columns: year, eta_cloglog) used to generate simulated epidemics")
    parser.add_argument("--check-optimizer", action="store_true",
                        help="Refit the empirical model ladder and every simulated candidate model with tighter optimisation and compare the results")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scales = parse_scales(args.scales)
    true_scales = parse_scales(args.true_scales)

    tables = run_empirical_analysis(root, scales, denominator_mode=args.denominator_mode, check_optimizer=args.check_optimizer)
    if not args.skip_simulation:
        tables.update(run_simulation_study(root, scales, true_scales, n_sim=args.n_sim, seed=args.seed,
                                           check_optimizer=args.check_optimizer,
                                           year_effects_file=(root / args.year_effects) if args.year_effects else None))

    write_tables(tables, output_dir)
    write_excel_workbook(tables, output_dir / "cssv_analysis_tables.xlsx")
    print(f"Wrote CSV tables to: {output_dir / 'csv'}")
    print(f"Wrote Excel workbook to: {output_dir / 'cssv_analysis_tables.xlsx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
