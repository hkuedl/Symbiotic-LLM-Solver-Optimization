"""Public exports for IS modules, whose directory name is a Python keyword."""

from importlib import import_module

_EXPORTS = {
    "IndustrialUncertaintyQuantifier": ("ours.is.uncertainty_is", "UncertaintyQuantifier"),
    "IndustrialOursSolver": ("ours.is.ours_is", "OursSolver"),
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
