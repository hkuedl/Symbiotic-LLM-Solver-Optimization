"""Public exports for IS modules, whose directory name is a Python keyword."""

from importlib import import_module

_EXPORTS = {
    "TabuSearchSolverIS": ("baselines.is.tabu_is", "TabuSearchSolver"),
    "SimulatedAnnealingSolverIS": ("baselines.is.sa_is", "SimulatedAnnealingSolver"),
    "RollingHorizonSolverIS": ("baselines.is.rh_is", "RollingHorizonSolver"),
    "LagrangianRelaxationSolverIS": ("baselines.is.lr_is", "LagrangianRelaxationSolver"),
    "LocalSearchSolverIS": ("baselines.is.lns_is", "LocalSearchSolver"),
    "FNOSolverIS": ("baselines.is.fno_is", "FNOSolver"),
    "LLMConfidenceSolverIS": ("baselines.is.confidence_llm_is", "LLMConfidenceSolver"),
    "LLMSolverIS": ("baselines.is.complete_llm_is", "LLMSolver"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    """Import an IS component lazily while keeping normal Python import syntax."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
