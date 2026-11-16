"""Daily experiment orchestration and result persistence."""

import os
import time

import pandas as pd

from baselines import FNOSolverIS as FNOSolver
from baselines import LagrangianRelaxationSolverIS as LagrangianRelaxationSolver
from baselines import LLMConfidenceSolverIS as LLMConfidenceSolver
from baselines import LLMSolverIS as LLMSolver
from baselines import LocalSearchSolverIS as LocalSearchSolver
from baselines import RollingHorizonSolverIS as RollingHorizonSolver
from baselines import SimulatedAnnealingSolverIS as SimulatedAnnealingSolver
from baselines import TabuSearchSolverIS as TabuSearchSolver
from common.paths import DATA_ROOT, PROJECT_ROOT
from common.results import append_summary, case_slug
from common.solver_config import DEFAULT_TIME_LIMIT, MIP_GAP, validate_time_limit
from models import IndustrialModel
from ours import IndustrialOursSolver as OursSolver
from solver import IndustrialSolver as Solver


class IndustrialSchedulingRunner:
    """IndustrialSchedulingRunner: daily experiment orchestration and result persistence."""

    def __init__(self, days, params, data_dir=None, output_dir=None, time_limit=DEFAULT_TIME_LIMIT):
        self.time_limit = validate_time_limit(time_limit)
        self.days = days
        self.params = params
        self.output_dir = output_dir if output_dir is not None else PROJECT_ROOT

        data_dir = data_dir if data_dir is not None else DATA_ROOT / "is"

        price_df = pd.read_csv(os.path.join(data_dir, "lmp_price_is.csv")).sort_values(
            ["day", "step"]
        )
        pv_df = pd.read_csv(os.path.join(data_dir, "pv_generation_is.csv")).sort_values(
            ["day", "step"]
        )
        order_df = pd.read_csv(os.path.join(data_dir, "production_order_is.csv")).sort_values("day")

        self.price_list = [
            price_df[price_df["day"] == d].sort_values("step")["price"].tolist() for d in days
        ]
        self.pv_list = [
            pv_df[pv_df["day"] == d].sort_values("step")["power"].tolist() for d in days
        ]
        self.q_req_list = [
            float(order_df[order_df["day"] == d]["production"].iloc[0]) for d in days
        ]

    def _paths(self, method):
        """Return detailed and summary CSV paths for this case and method."""
        name = case_slug(self.params.name)
        return (
            os.path.join(self.output_dir, "results", "is", name, f"{method}.csv"),
            os.path.join(self.output_dir, "record", "is", name, f"{method}.csv"),
        )

    def _print_solution(self, sol, q_req, solve_time):
        print(f"  Objective      = ${sol['objective']:,.2f}")
        print(f"  Production     = {sol['total_production']:,.2f} ton (Required: {q_req:,.2f})")
        print(f"  Solve time     = {solve_time:.1f}s")

    def _save_day(self, day, price, pv_output, q_req, p, sol, save_path):
        rows = []
        for t in range(p.num_steps):
            row = {
                "day": day,
                "step": t,
                "price": price[t],
                "pv_available": pv_output[t],
                "q_req": q_req,
                "P_grid": sol["P_grid"][t],
                "P_pv": sol["P_pv"][t],
                "P_ch": sol["P_ch"][t],
                "P_dis": sol["P_dis"][t],
                "SoC": sol["SoC"][t],
                "P_prod_total": sol["P_prod_total"][t],
            }
            for l in range(p.num_lines):
                row[f"Y_line{l}"] = sol["Y"][l]
                for k in range(p.num_processes):
                    row[f"u_L{l}K{k}"] = sol["u"][l][k][t]
                    row[f"prod_out_L{l}K{k}"] = sol["prod_out"][l][k][t]
            rows.append(row)
        df = pd.DataFrame(rows)
        if not os.path.exists(save_path):
            df.to_csv(save_path, index=False)
        else:
            df.to_csv(save_path, mode="a", header=False, index=False)

    def _init_save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)

    def _save_summary(self, row, save_path):
        """Align variable iteration fields before adding a daily summary."""
        row = {
            "case": self.params.name,
            "solver": "gurobi",
            "time_limit": self.time_limit,
            "mip_gap": MIP_GAP,
            **getattr(self, "run_metadata", {}),
            **row,
        }
        if row.get("status") == "infeasible":
            row["status"] = "no_solution"
        elif row.get("status") == "optimal":
            row["status"] = "feasible"
        append_summary(row, save_path)

    def run_solver(self):
        """Run the standalone solver with the original solver options."""
        save_path, summary_path = self._paths("solver")
        self._init_save(save_path)
        self._init_save(summary_path)
        model_obj = IndustrialModel(self.params)
        solver = Solver(time_limit=self.time_limit)
        for i, day in enumerate(self.days):
            price, pv, q_req = self.price_list[i], self.pv_list[i], self.q_req_list[i]
            print(f"\n{'=' * 60}")
            print(f"[Solver] Day {day}")
            print("=" * 60)
            model_obj.build(price, pv, q_req)
            t0 = time.time()
            sol = solver.solve(model_obj)
            solve_time = time.time() - t0
            if sol is None:
                print("Infeasible")
                self._save_summary(
                    {
                        "day": day,
                        "status": "infeasible",
                        "objective": None,
                        "production": None,
                        "solve_time": round(solve_time, 1),
                    },
                    summary_path,
                )
                continue
            self._print_solution(sol, q_req, solve_time)
            self._save_day(day, price, pv, q_req, self.params, sol, save_path)
            self._save_summary(
                {
                    "day": day,
                    "status": "optimal",
                    "objective": sol["objective"],
                    "production": sol["total_production"],
                    "solve_time": round(solve_time, 1),
                },
                summary_path,
            )
        print(f"\n[Solver] Done. Saved to: {save_path}")

    def run_fno(self, time_window=24, max_rounds=3):
        save_path, summary_path = self._paths("fno")
        self._init_save(save_path)
        self._init_save(summary_path)
        model_obj = IndustrialModel(self.params)
        solver = FNOSolver(
            time_window=time_window, max_rounds=max_rounds, time_limit=self.time_limit
        )
        for i, day in enumerate(self.days):
            price, pv, q_req = self.price_list[i], self.pv_list[i], self.q_req_list[i]
            print(f"\n{'=' * 60}")
            print(f"[FNO] Day {day}")
            print("=" * 60)
            model_obj.build(price, pv, q_req)
            t0 = time.time()
            sol = solver.solve(model_obj)
            total_time = time.time() - t0
            if sol is None:
                print("  Infeasible")
                self._save_summary(
                    {
                        "day": day,
                        "status": "infeasible",
                        "objective": None,
                        "production": None,
                        "total_time": round(total_time, 1),
                    },
                    summary_path,
                )
                continue
            self._print_solution(sol, q_req, total_time)
            self._save_day(day, price, pv, q_req, self.params, sol, save_path)
            self._save_summary(
                {
                    "day": day,
                    "status": "feasible",
                    "objective": sol["objective"],
                    "production": sol["total_production"],
                    "total_time": round(total_time, 1),
                },
                summary_path,
            )
        print(f"\n[FNO] Done. Saved to: {save_path}")

    def run_llm(self, api_key=None, base_url=None, llm_model="gpt-4o"):
        save_path, summary_path = self._paths("llm")
        self._init_save(save_path)
        self._init_save(summary_path)
        model_obj = IndustrialModel(self.params)
        solver = LLMSolver(
            api_key=api_key, base_url=base_url, model=llm_model, time_limit=self.time_limit
        )
        for i, day in enumerate(self.days):
            price, pv, q_req = self.price_list[i], self.pv_list[i], self.q_req_list[i]
            print(f"\n{'=' * 60}")
            print(f"[LLM] Day {day}")
            print("=" * 60)
            model_obj.build(price, pv, q_req)
            t0 = time.time()
            sol = solver.solve(model_obj, price, pv, q_req)
            total_time = time.time() - t0
            summary = {
                "day": day,
                "llm_time": round(solver.llm_time, 1),
                "solve_time": round(total_time - solver.llm_time, 1),
                "total_time": round(total_time, 1),
                "tokens_prompt": solver.token_usage.prompt_tokens if solver.token_usage else 0,
                "tokens_completion": solver.token_usage.completion_tokens
                if solver.token_usage
                else 0,
            }
            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None, "production": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, q_req, total_time)
            print(f"  LLM call time = {solver.llm_time:.1f}s")
            self._save_day(day, price, pv, q_req, self.params, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "production": sol["total_production"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[LLM] Done. Saved to: {save_path}")

    def run_llm_confidence(
        self, api_key=None, base_url=None, llm_model="gpt-4o", confidence_threshold=0.8
    ):
        save_path, summary_path = self._paths("llm-confidence")
        self._init_save(save_path)
        self._init_save(summary_path)
        model_obj = IndustrialModel(self.params)
        solver = LLMConfidenceSolver(
            api_key=api_key,
            base_url=base_url,
            model=llm_model,
            confidence_threshold=confidence_threshold,
            time_limit=self.time_limit,
        )
        for i, day in enumerate(self.days):
            price, pv, q_req = self.price_list[i], self.pv_list[i], self.q_req_list[i]
            print(f"\n{'=' * 60}\n[LLM-Conf] Day {day}  |  q_req = {q_req}\n{'=' * 60}")
            model_obj.build(price, pv, q_req)
            t0 = time.time()
            sol = solver.solve(model_obj, price, pv, q_req)
            total_time = time.time() - t0
            summary = {
                "day": day,
                "threshold": confidence_threshold,
                "llm_time": round(solver.llm_time, 1),
                "solve_time": round(total_time - solver.llm_time, 1),
                "total_time": round(total_time, 1),
                "fixed_count": solver.fix_stats.get("fixed", 0),
                "free_count": solver.fix_stats.get("free", 0),
                "tokens_prompt": solver.token_usage.prompt_tokens if solver.token_usage else 0,
                "tokens_completion": solver.token_usage.completion_tokens
                if solver.token_usage
                else 0,
            }
            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None, "production": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, q_req, total_time)
            print(f"  LLM call time  = {solver.llm_time:.1f}s")
            print(f"  Fixed/Free     = {solver.fix_stats['fixed']}/{solver.fix_stats['free']}")
            self._save_day(day, price, pv, q_req, self.params, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "production": sol["total_production"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[LLM-Conf] Done. Saved to: {save_path}")

    def run_ours(
        self,
        api_key=None,
        base_url=None,
        llm_model="gpt-4o",
        n_samples=5,
        strategy="epistemic_threshold",
        strategy_params=None,
        max_refine_iter=5,
    ):
        """Run Ours with the original sampling, fixing, and refinement defaults."""
        save_path, summary_path = self._paths("ours")
        self._init_save(save_path)
        self._init_save(summary_path)
        model_obj = IndustrialModel(self.params)
        solver = OursSolver(
            api_key=api_key,
            base_url=base_url,
            model=llm_model,
            n_samples=n_samples,
            strategy=strategy,
            strategy_params=strategy_params or {"tau_I": 0.1},
            max_refine_iter=max_refine_iter,
            time_limit=self.time_limit,
        )
        for i, day in enumerate(self.days):
            price, pv, q_req = self.price_list[i], self.pv_list[i], self.q_req_list[i]
            print(f"\n{'=' * 60}\n[Ours] Day {day}  |  q_req = {q_req}\n{'=' * 60}")
            model_obj.build(price, pv, q_req)
            t0 = time.time()
            sol = solver.solve(model_obj, price, pv, q_req)
            total_time = time.time() - t0
            summary = {
                "day": day,
                "model": llm_model,
                "n_samples": n_samples,
                "strategy": strategy,
                "sampling_time": round(solver.llm_time, 1),
                "n_iterations": len(solver.history),
                "total_time": round(total_time, 1),
                "tokens_prompt": solver.token_usage["prompt"],
                "tokens_completion": solver.token_usage["completion"],
            }
            for t_info in solver.iter_timings:
                it = t_info["iter"]
                summary[f"iter{it}_status"] = t_info["status"]
                summary[f"iter{it}_solve_time"] = round(t_info.get("solve_time", 0), 1)
                summary[f"iter{it}_llm_time"] = round(t_info.get("llm_refine_time", 0), 1)
                summary[f"iter{it}_obj"] = t_info.get("objective", None)
                summary[f"iter{it}_n_changed"] = t_info.get("n_changed", 0)
            if sol is None:
                print("  No feasible solution")
                summary.update({"status": "infeasible", "objective": None, "production": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, q_req, total_time)
            print(f"  LLM sampling   = {solver.llm_time:.1f}s")
            print(
                f"  Tokens         = prompt:{solver.token_usage['prompt']}, "
                f"completion:{solver.token_usage['completion']}"
            )
            print(f"  Iterations     = {len(solver.history)}")
            self._save_day(day, price, pv, q_req, self.params, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "production": sol["total_production"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[Ours] Done. Saved to: {save_path}")

    def run_sa(self, max_iter=200, T=5000.0, alpha=0.95, perturb_ratio=0.15):
        save_path, summary_path = self._paths("sa")
        self._init_save(save_path)
        self._init_save(summary_path)

        for i, day in enumerate(self.days):
            price, pv, q_req = self.price_list[i], self.pv_list[i], self.q_req_list[i]
            print(f"\n{'=' * 60}\n[SA] Day {day}  |  q_req = {q_req}\n{'=' * 60}")

            model_obj = IndustrialModel(self.params)
            model_obj.build(price, pv, q_req)

            solver = SimulatedAnnealingSolver(
                max_iter=max_iter,
                T0=T,
                alpha=alpha,
                perturb_ratio=perturb_ratio,
                time_limit=self.time_limit,
            )

            t0 = time.time()
            sol = solver.solve(model_obj, price, pv, q_req)
            total_time = time.time() - t0

            summary = {
                "day": day,
                "max_iter": max_iter,
                "T0": T,
                "alpha": alpha,
                "perturb_ratio": perturb_ratio,
                "n_iters": len(solver.history),
                "total_time": round(total_time, 1),
            }

            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None, "production": None})
                self._save_summary(summary, summary_path)
                continue

            self._print_solution(sol, q_req, total_time)
            self._save_day(day, price, pv, q_req, self.params, sol, save_path)

            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "production": sol["total_production"],
                }
            )
            self._save_summary(summary, summary_path)

        print(f"\n[SA] Done. Saved to: {save_path}")

    def run_lns(self, max_iter=30, destroy_ratio=0.25):
        save_path, summary_path = self._paths("lns")
        self._init_save(save_path)
        self._init_save(summary_path)
        for i, day in enumerate(self.days):
            price, pv, q_req = self.price_list[i], self.pv_list[i], self.q_req_list[i]
            print(f"\n{'=' * 60}\n[LNS] Day {day}  |  q_req = {q_req}\n{'=' * 60}")
            model_obj = IndustrialModel(self.params)
            model_obj.build(price, pv, q_req)
            solver = LocalSearchSolver(
                max_iter=max_iter, destroy_ratio=destroy_ratio, time_limit=self.time_limit
            )
            t0 = time.time()
            sol = solver.solve(model_obj, price, pv, q_req)
            total_time = time.time() - t0
            summary = {
                "day": day,
                "max_iter": max_iter,
                "destroy_ratio": destroy_ratio,
                "n_iters": len(solver.history),
                "total_time": round(total_time, 1),
            }
            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None, "production": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, q_req, total_time)
            self._save_day(day, price, pv, q_req, self.params, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "production": sol["total_production"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[LNS] Done. Saved to: {save_path}")

    def run_lr(self, max_iter=30, step_init=0.5, step_decay=0.95, recovery_interval=3):
        save_path, summary_path = self._paths("lr")
        self._init_save(save_path)
        self._init_save(summary_path)

        for i, day in enumerate(self.days):
            price = self.price_list[i]
            pv = self.pv_list[i]
            q_req = self.q_req_list[i]

            print(f"\n{'=' * 60}\n[LR] Day {day}  |  q_req = {q_req}\n{'=' * 60}")

            model_obj = IndustrialModel(self.params)
            model_obj.build(price, pv, q_req)

            solver = LagrangianRelaxationSolver(
                max_iter=max_iter,
                step_init=step_init,
                step_decay=step_decay,
                recovery_interval=recovery_interval,
                recovery_fix_ratios=(0.9, 0.7, 0.5),
                seed=42,
                time_limit=self.time_limit,
            )

            start = time.time()
            sol = solver.solve(model_obj, price, pv, q_req)
            total_time = time.time() - start

            summary = {
                "day": day,
                "max_iter": max_iter,
                "step_init": step_init,
                "step_decay": step_decay,
                "recovery_interval": recovery_interval,
                "n_iters": len(solver.history),
                "total_time": round(total_time, 1),
            }

            if sol is None:
                print("  Infeasible")
                summary.update(
                    {
                        "status": "infeasible",
                        "objective": None,
                        "production": None,
                    }
                )
                self._save_summary(summary, summary_path)
                continue

            self._print_solution(sol, q_req, total_time)
            self._save_day(
                day,
                price,
                pv,
                q_req,
                self.params,
                sol,
                save_path,
            )

            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "production": sol["total_production"],
                }
            )
            self._save_summary(summary, summary_path)

        print(f"\n[LR] Done. Saved to: {save_path}")

    def run_rh(self, horizon_length=24):
        save_path, summary_path = self._paths("rh")
        self._init_save(save_path)
        self._init_save(summary_path)
        for i, day in enumerate(self.days):
            price, pv, q_req = self.price_list[i], self.pv_list[i], self.q_req_list[i]
            print(f"\n{'=' * 60}\n[RH] Day {day}  |  q_req = {q_req}\n{'=' * 60}")
            model_obj = IndustrialModel(self.params)
            model_obj.build(price, pv, q_req)
            solver = RollingHorizonSolver(horizon_length=horizon_length, time_limit=self.time_limit)
            t0 = time.time()
            sol = solver.solve(model_obj, price, pv, q_req)
            total_time = time.time() - t0
            summary = {
                "day": day,
                "horizon_length": horizon_length,
                "n_windows": len(solver.history),
                "total_time": round(total_time, 1),
            }
            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None, "production": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, q_req, total_time)
            self._save_day(day, price, pv, q_req, self.params, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "production": sol["total_production"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[RH] Done. Saved to: {save_path}")

    def run_tabu(
        self, max_iter=200, tabu_tenure=5, n_candidates=5, perturb_ratio=0.15, diversify_interval=10
    ):
        save_path, summary_path = self._paths("tabu")
        self._init_save(save_path)
        self._init_save(summary_path)

        for i, day in enumerate(self.days):
            price, pv, q_req = self.price_list[i], self.pv_list[i], self.q_req_list[i]
            print(f"\n{'=' * 60}\n[Tabu] Day {day}  |  q_req = {q_req}\n{'=' * 60}")

            model_obj = IndustrialModel(self.params)
            model_obj.build(price, pv, q_req)

            solver = TabuSearchSolver(
                max_iter=max_iter,
                tabu_tenure=tabu_tenure,
                n_candidates=n_candidates,
                perturb_ratio=perturb_ratio,
                diversify_interval=diversify_interval,
                time_limit=self.time_limit,
            )

            t0 = time.time()
            sol = solver.solve(model_obj, price, pv, q_req)
            total_time = time.time() - t0

            summary = {
                "day": day,
                "max_iter": max_iter,
                "tabu_tenure": tabu_tenure,
                "n_candidates": n_candidates,
                "n_iters": len(solver.history),
                "total_time": round(total_time, 1),
            }

            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None, "production": None})
                self._save_summary(summary, summary_path)
                continue

            self._print_solution(sol, q_req, total_time)
            self._save_day(day, price, pv, q_req, self.params, sol, save_path)

            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "production": sol["total_production"],
                }
            )
            self._save_summary(summary, summary_path)

        print(f"\n[Tabu] Done. Saved to: {save_path}")


def main(argv=None):
    """Run the command-line entry point for IS experiments."""
    from runners.cli import run_cli

    return run_cli("is", argv)


if __name__ == "__main__":
    main()
