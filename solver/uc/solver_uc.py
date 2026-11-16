"""Standalone mathematical solver interface."""

import time

from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit


class Solver:
    """Solver: standalone mathematical solver interface."""

    def __init__(self, tee=True, threads=0, time_limit=DEFAULT_TIME_LIMIT):
        self.time_limit = validate_time_limit(time_limit)
        self.tee = tee
        self.threads = threads
        self.solve_time = 0

    def solve(self, model_obj):
        """Run the method and return the extracted solution, or None on failure."""
        m = model_obj.model
        if m is None:
            raise RuntimeError("Model not built yet.")

        solver = create_solver(self.time_limit)
        solver.options["DisplayInterval"] = 60
        solver.options["Method"] = 3
        if self.threads > 0:
            solver.options["Threads"] = self.threads

        t0 = time.time()
        result = solve_model(solver, m, tee=self.tee)
        self.solve_time = time.time() - t0

        tc = str(result.solver.termination_condition).lower()
        print(f"  [MILP] Done in {self.solve_time:.1f}s, condition={tc}")

        if tc not in ("optimal", "feasible", "locallyoptimal"):
            return None

        return model_obj.extract_results()
