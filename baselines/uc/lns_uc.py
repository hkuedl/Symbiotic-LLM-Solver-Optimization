"""Reference optimization method."""

import logging
import random
import time

import numpy as np
import pyomo.environ as pyo

from baselines.uc.lr_uc import LagrangianRelaxationSolver
from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit
from models.uc.model_uc import UnitCommitmentModel


class LargeNeighborhoodSearchSolver:
    """LargeNeighborhoodSearchSolver: reference optimization method."""

    def __init__(
        self,
        max_iter=50,
        destroy_ratio=0.08,
        lr_warmup_iters=25,
        no_improve_limit=15,
        seed=42,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.time_limit = validate_time_limit(time_limit)
        self.max_iter = max_iter
        self.destroy_ratio = destroy_ratio
        self.lr_warmup_iters = lr_warmup_iters
        self.no_improve_limit = no_improve_limit
        self.seed = seed
        self.solve_time = 0.0
        self.history = []

    def solve(self, model_obj, demand_bus, reserve, wind=None, solar=None):
        """Run the method and return the extracted solution, or None on failure."""
        p = model_obj.p
        if model_obj.model is None:
            raise RuntimeError("Model not built yet.")

        total_start = time.time()
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.history = []

        G, T = p.num_generators, p.num_steps
        total_vars = G * T

        self._p = p
        self._demand_bus = demand_bus
        self._reserve = reserve
        self._wind = wind
        self._solar = solar

        demand_total = np.array(
            [sum(demand_bus[n][t] for n in range(p.num_buses)) for t in range(T)]
        )
        self._demand_total = demand_total

        self._base_model_obj = UnitCommitmentModel(p)
        self._base_m = self._base_model_obj.build(demand_bus, reserve, wind, solar)
        self._solver = create_solver(self.time_limit)
        self._solver.options["MIPFocus"] = 1
        self._solver.options["Seed"] = self.seed

        # Phase 1: LR warmup
        print(f"[LNS] Phase 1: LR warmup ({self.lr_warmup_iters} iters)", flush=True)
        best_u, best_obj, best_sol = self._get_initial_solution(
            model_obj, demand_bus, reserve, wind, solar
        )

        if best_sol is None:
            print("[LNS] Failed to obtain initial feasible solution.", flush=True)
            self._cleanup()
            self.solve_time = time.time() - total_start
            return None

        n_destroy = max(1, int(total_vars * self.destroy_ratio))

        self.history.append(
            {
                "iter": 0,
                "obj": best_obj,
                "best_obj": best_obj,
                "status": "initial",
            }
        )
        print(f"[LNS] Initial obj=${best_obj:,.0f}", flush=True)
        print(
            f"[LNS] Phase 2: Destroy & Repair "
            f"(max_iter={self.max_iter}, "
            f"destroy={n_destroy}/{total_vars}, "
            f"ratio={self.destroy_ratio:.1%})",
            flush=True,
        )

        no_improve = 0

        strategy_idx = 0

        for iteration in range(1, self.max_iter + 1):
            free_set = self._destroy(best_u, p, n_destroy, strategy_idx)
            strategy_idx = (strategy_idx + 1) % 4

            t0 = time.time()
            new_obj, new_sol, new_u = self._repair(best_u, free_set, p)
            repair_time = time.time() - t0

            if new_sol is not None and new_obj < best_obj - 1e-2:
                improvement = best_obj - new_obj
                best_obj = new_obj
                best_u = new_u.copy()
                best_sol = new_sol
                no_improve = 0
                status = "improved"
                print(
                    f"  Iter {iteration}: obj=${best_obj:,.0f} "
                    f"(↓${improvement:,.0f}), "
                    f"free={len(free_set)}, "
                    f"time={repair_time:.1f}s",
                    flush=True,
                )
            else:
                no_improve += 1
                status = "infeasible" if new_sol is None else "no_improve"
                if iteration % 5 == 0 or new_sol is None:
                    print(
                        f"  Iter {iteration}: {status}, "
                        f"stall={no_improve}, "
                        f"free={len(free_set)}, "
                        f"time={repair_time:.1f}s",
                        flush=True,
                    )

            self.history.append(
                {
                    "iter": iteration,
                    "obj": new_obj if new_sol else None,
                    "best_obj": best_obj,
                    "status": status,
                    "time": repair_time,
                    "n_free": len(free_set),
                }
            )

            if no_improve >= self.no_improve_limit:
                print(f"  Converged at iter {iteration} (stall={no_improve}).", flush=True)
                break

        self._cleanup()
        self.solve_time = time.time() - total_start
        print(
            f"[LNS] Done. obj=${best_obj:,.0f}, "
            f"iters={len(self.history) - 1}, "
            f"time={self.solve_time:.1f}s",
            flush=True,
        )
        return best_sol

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------
    def _cleanup(self):
        self._base_model_obj = None
        self._base_m = None
        self._solver = None

    # ------------------------------------------------------------------
    #  Initial solution via LR warmup
    # ------------------------------------------------------------------
    def _get_initial_solution(self, model_obj, demand_bus, reserve, wind, solar):
        p = model_obj.p

        lr = LagrangianRelaxationSolver(seed=self.seed, time_limit=self.time_limit)
        best_u, best_ub = lr.warmup(
            model_obj,
            demand_bus,
            reserve,
            wind,
            solar,
            warmup_iters=self.lr_warmup_iters,
            recovery_interval=3,
            verbose=True,
        )

        if best_u is None or not np.isfinite(best_ub):
            return None, float("inf"), None

        obj, sol, u_fixed = self._repair(best_u, set(), self._p)

        if sol is None:
            return None, float("inf"), None

        return u_fixed, obj, sol

    # ------------------------------------------------------------------
    #  Structured destroy strategies
    # ------------------------------------------------------------------
    def _destroy(self, u, p, n_destroy, strategy_idx):
        G, T = p.num_generators, p.num_steps
        free_set = set()

        strategy = strategy_idx % 4

        if strategy == 0:
            n_gens = random.randint(1, min(3, G))
            gens = random.sample(range(G), n_gens)
            for g in gens:
                for t in range(T):
                    free_set.add((g, t))

        elif strategy == 1:
            window = random.randint(max(1, T // 6), max(1, T // 3))
            t_start = random.randint(0, max(0, T - window))
            n_gens = random.randint(max(1, G // 4), max(1, G // 2))
            gens = random.sample(range(G), n_gens)
            for g in gens:
                for t in range(t_start, min(t_start + window, T)):
                    free_set.add((g, t))

        elif strategy == 2:
            startup_count = np.zeros(G)
            for g in range(G):
                for t in range(1, T):
                    if u[g, t] != u[g, t - 1]:
                        startup_count[g] += 1

            n_gens = random.randint(2, min(5, G))
            top_gens = np.argsort(startup_count)[-n_gens * 2 :]
            gens = random.sample(list(top_gens), min(n_gens, len(top_gens)))
            for g in gens:
                for t in range(T):
                    free_set.add((g, t))

        else:
            all_keys = [(g, t) for g in range(G) for t in range(T)]
            free_set = set(random.sample(all_keys, min(n_destroy, len(all_keys))))

        max_free = int(G * T * 0.4)
        if len(free_set) > max_free:
            free_set = set(random.sample(list(free_set), max_free))
        if len(free_set) < 1:
            all_keys = [(g, t) for g in range(G) for t in range(T)]
            free_set = set(random.sample(all_keys, min(n_destroy, len(all_keys))))

        return free_set

    # ------------------------------------------------------------------
    #  Repair via sub-MIP (reuse pre-built model)
    # ------------------------------------------------------------------
    def _repair(self, u, free_set, p):
        G, T = p.num_generators, p.num_steps
        m = self._base_m

        for g in range(G):
            for t in range(T):
                m.u[g, t].unfix()
                m.v[g, t].unfix()
                m.w[g, t].unfix()

        for g in range(G):
            for t in range(T):
                val = int(u[g, t])
                if t == 0:
                    prev = p.u0[g]
                else:
                    prev = int(u[g, t - 1])

                if (g, t) not in free_set:
                    m.u[g, t].fix(val)

                    prev_is_fixed = (t == 0) or ((g, t - 1) not in free_set)
                    if prev_is_fixed:
                        m.v[g, t].fix(max(0, val - prev))
                        m.w[g, t].fix(max(0, prev - val))
                    else:
                        m.v[g, t].set_value(max(0, val - prev))
                        m.w[g, t].set_value(max(0, prev - val))
                else:
                    m.u[g, t].set_value(val)
                    m.v[g, t].set_value(max(0, val - prev))
                    m.w[g, t].set_value(max(0, prev - val))

        logger = logging.getLogger("pyomo.core")
        old_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            results = solve_model(self._solver, m, tee=False, warmstart=True)
        finally:
            logger.setLevel(old_level)

        status = str(results.solver.termination_condition).lower()
        has_solution = len(results.solution) > 0
        if status not in ("optimal", "feasible", "locallyoptimal") and has_solution:
            status = "feasible"

        if status not in ("optimal", "feasible", "locallyoptimal"):
            return float("inf"), None, None

        self._base_model_obj.model = m
        sol = self._base_model_obj.extract_results()

        new_u = np.zeros((G, T), dtype=int)
        for g in range(G):
            for t in range(T):
                new_u[g, t] = int(round(pyo.value(m.u[g, t])))

        return sol["objective"], sol, new_u
