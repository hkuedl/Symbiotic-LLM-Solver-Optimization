"""Standalone mathematical solver interface."""

from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit


class Solver:
    """Solver: standalone mathematical solver interface."""

    def __init__(self, tee=True, time_limit=DEFAULT_TIME_LIMIT):
        self.time_limit = validate_time_limit(time_limit)
        self.tee = tee
        self.solve_time = 0

    def solve(self, model_obj):
        """Run the method and return the extracted solution, or None on failure."""
        m = model_obj.model
        if m is None:
            raise RuntimeError("Model not built yet. Call model_obj.build() first.")

        solver = create_solver(self.time_limit)
        solver.options["DisplayInterval"] = 60

        result = solve_model(solver, m, tee=self.tee)

        tc = str(result.solver.termination_condition).lower()
        print(f"  [MILP] Done in {self.solve_time:.1f}s, condition={tc}")

        if tc not in ("optimal", "feasible", "locallyoptimal"):
            print(f"  [MILP] Infeasible: {tc}")
            return None

        sol = model_obj.extract_results()
        return sol
