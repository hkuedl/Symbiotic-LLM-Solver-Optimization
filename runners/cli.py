"""Command-line configuration shared by the UC and industrial runners."""

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
from pathlib import Path

from common.paths import PROJECT_ROOT
from common.solver_config import DEFAULT_TIME_LIMIT, MIP_GAP, validate_time_limit

METHODS = {
    "solver": "run_solver",
    "ours": "run_ours",
    "fno": "run_fno",
    "sa": "run_sa",
    "lns": "run_lns",
    "lr": "run_lr",
    "rh": "run_rh",
    "tabu": "run_tabu",
    "llm": "run_llm",
    "llm-confidence": "run_llm_confidence",
}
LLM_METHODS = {"ours", "llm", "llm-confidence"}


def parse_days(value):
    """Parse comma-separated day IDs or a Python-style half-open range."""
    try:
        if ":" in value:
            start, stop = (int(v) for v in value.split(":"))
            days = list(range(start, stop))
        else:
            days = [int(v) for v in value.split(",")]
        if not days or min(days) < 0 or len(set(days)) != len(days):
            raise ValueError
        return days
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use unique nonnegative IDs (0,1,2) or a half-open range (0:30)."
        ) from exc


def build_parser(problem):
    """Expose case selection and the existing runner method arguments."""
    parser = argparse.ArgumentParser(description=f"Run {problem.upper()} benchmarks.")
    cases = ("118-bus", "1354-bus") if problem == "uc" else ("cement-plant", "steel-powder-plant")
    parser.add_argument("--case", required=True, choices=cases)
    parser.add_argument("--method", choices=METHODS, default="solver")
    parser.add_argument("--days", type=parse_days, default=[0])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-dir", type=Path, help="Override the case data directory.")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help="Seconds per Gurobi call (default: 1e9); includes auxiliary solves.",
    )
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "gpt-4o"))
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--max-refine-iter", type=int)
    parser.add_argument(
        "--strategy", choices=("epistemic_threshold", "epistemic_percentile", "epistemic_budget")
    )
    parser.add_argument("--strategy-params", type=json.loads, help='JSON, e.g. {"tau_I": 0.1}.')
    parser.add_argument(
        "--method-options",
        type=json.loads,
        default={},
        help="JSON object of existing runner keyword arguments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate daily inputs and print configuration without solving.",
    )
    return parser


def validate_daily_inputs(runner, problem):
    """Reject missing or incomplete operational days before solver work starts."""
    expected = runner.params.num_steps
    for i, day in enumerate(runner.days):
        if problem == "uc":
            series = list(runner.demand_bus_list[i]) + [runner.reserve_list[i]]
            series += list((runner.wind_list[i] or {}).values())
            series += list((runner.solar_list[i] or {}).values())
        else:
            series = [runner.price_list[i], runner.pv_list[i]]
        if any(len(values) != expected for values in series):
            raise ValueError(f"Day {day} must have exactly {expected} steps in every input series.")


def run_cli(problem, argv=None):
    """Validate configuration and execute a method with CSV-only experiment records."""
    parser = build_parser(problem)
    args = parser.parse_args(argv)
    try:
        args.time_limit = validate_time_limit(args.time_limit)
    except ValueError as exc:
        parser.error(str(exc))
    if not isinstance(args.method_options, dict):
        parser.error("--method-options must be a JSON object.")
    if {"api_key", "base_url"} & args.method_options.keys():
        parser.error("Use OPENAI_API_KEY and OPENAI_BASE_URL environment variables.")
    if args.strategy_params is not None and not isinstance(args.strategy_params, dict):
        parser.error("--strategy-params must be a JSON object.")
    if any(v is not None and v < 1 for v in (args.n_samples, args.max_refine_iter)):
        parser.error("Sample and refinement counts must be positive.")

    if problem == "uc":
        from params.uc.params_uc import bus118, bus1354
        from runners.run_uc import UnitCommitmentRunner

        params = {"118-bus": bus118, "1354-bus": bus1354}[args.case]
        runner_type = UnitCommitmentRunner
    else:
        from params import cement_plant, steel_powder_plant
        from runners.run_is import IndustrialSchedulingRunner

        params = {"cement-plant": cement_plant, "steel-powder-plant": steel_powder_plant}[args.case]
        runner_type = IndustrialSchedulingRunner
    try:
        runner = runner_type(
            args.days,
            params,
            data_dir=args.data_dir,
            output_dir=args.output_dir.resolve(),
            time_limit=args.time_limit,
        )
        validate_daily_inputs(runner, problem)
    except (IndexError, KeyError, ValueError, OSError) as exc:
        parser.error(f"Cannot load the requested daily inputs: {exc}")

    method = getattr(runner, METHODS[args.method])
    options = dict(args.method_options)
    ours_options = {
        key: getattr(args, key)
        for key in ("n_samples", "max_refine_iter", "strategy", "strategy_params")
        if getattr(args, key) is not None
    }
    if ours_options and args.method != "ours":
        parser.error("Sampling, strategy, and refinement flags apply only to ours.")
    options.update(ours_options)
    if args.method in LLM_METHODS:
        options.setdefault("llm_model", args.llm_model)
        if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
            parser.error("Set OPENAI_API_KEY before running an LLM method.")
    try:
        bound = inspect.signature(method).bind(**options)
    except TypeError as exc:
        parser.error(str(exc))
    bound.apply_defaults()
    effective = {k: v for k, v in bound.arguments.items() if k not in {"api_key", "base_url"}}
    versions = {}
    for package in ("numpy", "pandas", "pyomo", "gurobipy", "openai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {
        "problem": problem,
        "case": params.name,
        "method": args.method,
        "days": args.days,
        "num_steps": params.num_steps,
        "dt_hours": params.dt,
        "method_options": effective,
        "python": platform.python_version(),
        "packages": versions,
        "solver": "gurobi",
        "time_limit": args.time_limit,
        "mip_gap": MIP_GAP,
    }
    print(json.dumps(metadata, indent=2))
    if args.dry_run:
        return metadata
    runner.run_metadata = {
        "method": args.method,
        "python_version": metadata["python"],
        **{f"version_{package}": version for package, version in versions.items()},
        **{f"config_{key}": value for key, value in effective.items()},
    }
    method(**options)
    return metadata
