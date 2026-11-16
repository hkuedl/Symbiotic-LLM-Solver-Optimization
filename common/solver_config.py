"""Shared Gurobi settings for main, auxiliary, and diagnostic solves."""

import math

import pyomo.environ as pyo

DEFAULT_TIME_LIMIT = 1e9
MIP_GAP = 1e-3


def validate_time_limit(time_limit):
    """Return a finite positive per-call limit, measured in seconds."""
    value = float(time_limit)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("time_limit must be a finite positive number of seconds.")
    return value


def create_solver(time_limit=DEFAULT_TIME_LIMIT):
    """Create a Pyomo Gurobi solver with the shared stopping settings."""
    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = validate_time_limit(time_limit)
    solver.options["MIPGap"] = MIP_GAP
    return solver


def configure_gurobi(model, time_limit=DEFAULT_TIME_LIMIT):
    """Apply the same settings to a native gurobipy diagnostic model."""
    model.Params.TimeLimit = validate_time_limit(time_limit)
    model.Params.MIPGap = MIP_GAP


def solve_model(solver, model, **kwargs):
    """Preserve a feasible incumbent when Gurobi stops at its time limit.

    Explicit loading avoids Pyomo errors when a short limit leaves no solution.
    Callers that request deferred loading still receive the solution container.
    """
    load_solutions = kwargs.pop("load_solutions", True)
    results = solver.solve(model, load_solutions=False, **kwargs)
    if len(results.solution) > 0:
        if results.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
            results.solver.termination_condition = pyo.TerminationCondition.feasible
        if load_solutions:
            model.solutions.load_from(results)
    return results
