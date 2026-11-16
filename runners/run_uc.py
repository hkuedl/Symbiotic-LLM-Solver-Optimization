"""Daily experiment orchestration and result persistence."""

import copy
import os
import time

import numpy as np
import pandas as pd

from baselines.uc.complete_llm_uc import LLMSolver
from baselines.uc.confidence_llm_uc import LLMConfidenceSolver
from baselines.uc.fno_uc import FNOSolver
from baselines.uc.lns_uc import LargeNeighborhoodSearchSolver
from baselines.uc.lr_uc import LagrangianRelaxationSolver
from baselines.uc.rh_uc import RollingHorizonSolver
from baselines.uc.sa_uc import SimulatedAnnealingSolver
from baselines.uc.tabu_uc import TabuSearchSolver
from common.paths import DATA_ROOT, PROJECT_ROOT
from common.results import append_summary, case_slug
from common.solver_config import DEFAULT_TIME_LIMIT, MIP_GAP, validate_time_limit
from models.uc.model_uc import UnitCommitmentModel
from ours.uc.ours_uc import OursSolver
from solver.uc.solver_uc import Solver


class UnitCommitmentRunner:
    """UnitCommitmentRunner: daily experiment orchestration and result persistence."""

    def __init__(self, days, params, data_dir=None, output_dir=None, time_limit=DEFAULT_TIME_LIMIT):
        self.time_limit = validate_time_limit(time_limit)
        self.days = days
        self.params = params
        self.output_dir = output_dir if output_dir is not None else PROJECT_ROOT

        name = params.name
        if data_dir is None:
            data_dir = DATA_ROOT / "uc" / name
        self.data_dir = data_dir

        N = params.num_buses
        T = params.num_steps

        bus_load_df = pd.read_csv(os.path.join(data_dir, "bus_load_uc.csv"))
        wind_df = pd.read_csv(os.path.join(data_dir, "wind_uc.csv"))
        solar_df = pd.read_csv(os.path.join(data_dir, "solar_uc.csv"))
        reserve_df = pd.read_csv(os.path.join(data_dir, "reserve_uc.csv"))

        bus_load_cols = [f"bus_{i}" for i in range(N)]
        wind_cols = sorted(
            [c for c in wind_df.columns if c.startswith("wind_bus_")],
            key=lambda c: int(c.replace("wind_bus_", "")),
        )
        solar_cols = sorted(
            [c for c in solar_df.columns if c.startswith("solar_bus_")],
            key=lambda c: int(c.replace("solar_bus_", "")),
        )

        self.demand_bus_list = []
        self.reserve_list = []
        self.wind_list = []
        self.solar_list = []

        for day in days:
            dl = bus_load_df[bus_load_df["day"] == day].sort_values("step")
            self.demand_bus_list.append([dl[col].tolist() for col in bus_load_cols])

            rl = reserve_df[reserve_df["day"] == day].sort_values("step")
            self.reserve_list.append(rl["reserve"].tolist())

            wl = wind_df[wind_df["day"] == day].sort_values("step")
            wind_dict = {}
            for col in wind_cols:
                idx = int(col.replace("wind_bus_", ""))
                wind_dict[idx] = wl[col].tolist()
            self.wind_list.append(wind_dict if wind_dict else None)

            sl = solar_df[solar_df["day"] == day].sort_values("step")
            solar_dict = {}
            for col in solar_cols:
                idx = int(col.replace("solar_bus_", ""))
                solar_dict[idx] = sl[col].tolist()
            self.solar_list.append(solar_dict if solar_dict else None)

    def _paths(self, method):
        """Return detailed and summary CSV paths for this case and method."""
        name = case_slug(self.params.name)
        return (
            os.path.join(self.output_dir, "results", "uc", name, f"{method}.csv"),
            os.path.join(self.output_dir, "record", "uc", name, f"{method}.csv"),
        )

    def _print_solution(self, sol, solve_time):
        print(f"  Objective      = ${sol['objective']:,.2f}")
        print(f"  Total gen      = {sum(sol['total_gen']):,.1f} MWh")
        print(f"  Total demand   = {sum(sol['demand']):,.1f} MWh")
        if "wind_curt_total" in sol:
            total_curt = sum(sol["wind_curt_total"]) + sum(sol.get("solar_curt_total", []))
            print(f"  Curtailment    = {total_curt:,.1f} MWh")
        print(f"  Solve time     = {solve_time:.1f}s")

    def _save_day(self, day, sol, save_path):
        p = self.params
        rows = []
        for t in range(p.num_steps):
            row = {
                "day": day,
                "step": t,
                "demand": sol["demand"][t],
                "total_gen": sol["total_gen"][t],
            }
            for g in range(p.num_generators):
                row[f"u_G{g}"] = sol["u"][g][t]
                row[f"P_G{g}"] = sol["P"][g][t]
            if "P_wind" in sol:
                for bus_idx in sol["wind_buses"]:
                    row[f"P_wind_bus{bus_idx + 1}"] = sol["P_wind"][bus_idx][t]
                    row[f"wind_curt_bus{bus_idx + 1}"] = sol["wind_curt"][bus_idx][t]
                row["P_wind_total"] = sol["P_wind_total"][t]
                row["wind_curt_total"] = sol["wind_curt_total"][t]
            if "P_solar" in sol:
                for bus_idx in sol["solar_buses"]:
                    row[f"P_solar_bus{bus_idx + 1}"] = sol["P_solar"][bus_idx][t]
                    row[f"solar_curt_bus{bus_idx + 1}"] = sol["solar_curt"][bus_idx][t]
                row["P_solar_total"] = sol["P_solar_total"][t]
                row["solar_curt_total"] = sol["solar_curt_total"][t]
            if "P_ch" in sol:
                row["P_ch"] = sol["P_ch"][t]
                row["P_dis"] = sol["P_dis"][t]
                row["SoC"] = sol["SoC"][t]
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

    def _build_day(self, i):
        day_params = copy.deepcopy(self.params)
        self._set_initial_state(day_params, i)
        model_obj = UnitCommitmentModel(day_params)
        model_obj.build(
            self.demand_bus_list[i],
            self.reserve_list[i],
            self.wind_list[i],
            self.solar_list[i],
        )
        return model_obj

    def _set_initial_state(self, p, i):
        """Set the day-specific initial commitment using the original heuristic."""
        demand_bus = self.demand_bus_list[i]
        reserve = self.reserve_list[i]
        wind = self.wind_list[i]
        solar = self.solar_list[i]

        demand_t0 = sum(demand_bus[n][0] for n in range(p.num_buses))
        reserve_t0 = reserve[0]
        re_t0 = 0
        if wind:
            re_t0 += sum(wind[b][0] for b in wind.keys())
        if solar:
            re_t0 += sum(solar[b][0] for b in solar.keys())

        gen_load = max(demand_t0 - re_t0, 0)
        reserve_need = demand_t0 + reserve_t0

        day = self.days[i]
        rng = np.random.RandomState(day * 31 + 42)
        gen_order = sorted(range(p.num_generators), key=lambda g: p.cost_b[g])

        online_list = []
        total_p0 = 0.0
        total_pmax_on = 0.0

        for g in gen_order:
            p0_g = round(rng.uniform(p.P_min[g], p.P_max[g]), 4)
            online_list.append((g, p0_g))
            total_p0 += p0_g
            total_pmax_on += p.P_max[g]

            if total_p0 >= gen_load and total_pmax_on >= reserve_need:
                break

        if total_p0 > gen_load and len(online_list) > 0:
            last_g, last_p0 = online_list[-1]
            overshoot = total_p0 - gen_load
            adjusted = max(last_p0 - overshoot, p.P_min[last_g])
            online_list[-1] = (last_g, round(adjusted, 4))

        online_set = set(g for g, _ in online_list)
        p0_map = {g: p0 for g, p0 in online_list}
        far_steps = max(p.min_up_time) * 10

        for g in range(p.num_generators):
            if g in online_set:
                p.u0[g] = 1
                raw_p0 = p0_map[g]
                max_safe = min(p.P_max[g], p.shutdown_ramp[g])
                p.P0[g] = round(max(min(raw_p0, max_safe), p.P_min[g]), 4)
                p.initial_status_steps[g] = far_steps
            else:
                p.u0[g] = 0
                p.P0[g] = 0.0
                p.initial_status_steps[g] = -far_steps

    def _print_line_utilization(self, model_obj):
        util = model_obj.line_utilization()
        n_congested_80 = sum(1 for v in util.values() if v["utilization"] > 0.80)
        n_congested_95 = sum(1 for v in util.values() if v["utilization"] > 0.95)
        max_util = max(v["utilization"] for v in util.values()) if util else 0
        print(
            f"  Lines >80%: {n_congested_80}/{len(util)}, "
            f">95%: {n_congested_95}/{len(util)}, "
            f"max: {max_util:.1%}"
        )
        top5 = sorted(util.items(), key=lambda x: x[1]["utilization"], reverse=True)[:5]
        for k, info in top5:
            print(
                f"    L{k} (bus {info['from']}→{info['to']}): "
                f"{info['utilization']:.1%}, "
                f"max_flow={info['max_flow']:.1f}/{info['capacity']:.0f} MW"
            )
        return util

    def run_solver(self):
        """Run the standalone solver with the original solver options."""
        save_path, summary_path = self._paths("solver")
        self._init_save(save_path)
        self._init_save(summary_path)
        solver = Solver(time_limit=self.time_limit)

        for i, day in enumerate(self.days):
            print(f"\n{'=' * 60}")
            print(f"[Solver] Day {day}")
            print("=" * 60)
            model_obj = self._build_day(i)
            t0 = time.time()
            sol = solver.solve(model_obj)
            solve_time = time.time() - t0
            if sol is None:
                print("  Infeasible")
                self._save_summary(
                    {
                        "day": day,
                        "status": "infeasible",
                        "objective": None,
                        "solve_time": round(solve_time, 1),
                    },
                    summary_path,
                )
                continue
            self._print_solution(sol, solve_time)
            costs = model_obj.cost_breakdown()
            self._print_line_utilization(model_obj)
            self._save_day(day, sol, save_path)
            self._save_summary(
                {
                    "day": day,
                    "status": "optimal",
                    "objective": sol["objective"],
                    "fuel_cost": costs["fuel_cost"],
                    "startup_cost": costs["startup_cost"],
                    "shutdown_cost": costs["shutdown_cost"],
                    "curtailment_cost": costs["curtailment_cost"],
                    "total_cost": costs["total_cost"],
                    "solve_time": round(solve_time, 1),
                },
                summary_path,
            )
        print(f"\n[Solver] Done. Saved to: {save_path}")

    def run_llm(self, api_key=None, base_url=None, llm_model="gpt-4o"):
        save_path, summary_path = self._paths("llm")
        self._init_save(save_path)
        self._init_save(summary_path)
        solver = LLMSolver(
            api_key=api_key, base_url=base_url, model=llm_model, time_limit=self.time_limit
        )

        for i, day in enumerate(self.days):
            demand_bus = self.demand_bus_list[i]
            reserve = self.reserve_list[i]
            wind = self.wind_list[i]
            solar = self.solar_list[i]
            print(f"\n{'=' * 60}")
            print(f"[LLM] Day {day}")
            print("=" * 60)
            model_obj = self._build_day(i)
            t0 = time.time()
            sol = solver.solve(model_obj, demand_bus, reserve, wind, solar)
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
                summary.update({"status": "infeasible", "objective": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, total_time)
            print(f"  LLM call time = {solver.llm_time:.1f}s")
            self._save_day(day, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
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
        solver = LLMConfidenceSolver(
            api_key=api_key,
            base_url=base_url,
            model=llm_model,
            confidence_threshold=confidence_threshold,
            time_limit=self.time_limit,
        )

        for i, day in enumerate(self.days):
            demand_bus = self.demand_bus_list[i]
            reserve = self.reserve_list[i]
            wind = self.wind_list[i]
            solar = self.solar_list[i]
            print(f"\n{'=' * 60}\n[LLM-Conf] Day {day}\n{'=' * 60}")
            model_obj = self._build_day(i)
            t0 = time.time()
            sol = solver.solve(model_obj, demand_bus, reserve, wind, solar)
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
                summary.update({"status": "infeasible", "objective": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, total_time)
            print(f"  LLM call time  = {solver.llm_time:.1f}s")
            print(f"  Fixed/Free     = {solver.fix_stats['fixed']}/{solver.fix_stats['free']}")
            self._save_day(day, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
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
            demand_bus = self.demand_bus_list[i]
            reserve = self.reserve_list[i]
            wind = self.wind_list[i]
            solar = self.solar_list[i]
            print(f"\n{'=' * 60}\n[Ours] Day {day}\n{'=' * 60}")
            model_obj = self._build_day(i)
            t0 = time.time()
            sol = solver.solve(model_obj, demand_bus, reserve, wind, solar)
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
                summary.update({"status": "infeasible", "objective": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, total_time)
            print(f"  LLM sampling   = {solver.llm_time:.1f}s")
            print(
                f"  Tokens         = prompt:{solver.token_usage['prompt']}, "
                f"completion:{solver.token_usage['completion']}"
            )
            print(f"  Iterations     = {len(solver.history)}")
            self._save_day(day, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[Ours] Done. Saved to: {save_path}")

    def run_fno(self, time_window=24, max_rounds=3, lr_warmup_iters=20):
        save_path, summary_path = self._paths("fno")
        self._init_save(save_path)
        self._init_save(summary_path)
        for i, day in enumerate(self.days):
            print(f"\n{'=' * 60}\n[FNO] Day {day}\n{'=' * 60}")
            model_obj = self._build_day(i)
            demand_bus = self.demand_bus_list[i]
            reserve = self.reserve_list[i]
            wind = self.wind_list[i]
            solar = self.solar_list[i]
            N, T = self.params.num_buses, self.params.num_steps
            demand_total = [sum(demand_bus[n][t] for n in range(N)) for t in range(T)]
            solver = FNOSolver(
                time_window=time_window,
                max_rounds=max_rounds,
                lr_warmup_iters=lr_warmup_iters,
                time_limit=self.time_limit,
            )
            t0 = time.time()
            sol = solver.solve(model_obj, demand_bus, demand_total, reserve, wind=wind, solar=solar)
            total_time = time.time() - t0
            summary = {
                "day": day,
                "time_window": time_window,
                "max_rounds": max_rounds,
                "lr_warmup_iters": lr_warmup_iters,
                "total_time": round(total_time, 1),
            }
            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, total_time)
            costs = model_obj.cost_breakdown()
            self._print_line_utilization(model_obj)
            self._save_day(day, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "fuel_cost": costs["fuel_cost"],
                    "startup_cost": costs["startup_cost"],
                    "shutdown_cost": costs["shutdown_cost"],
                    "curtailment_cost": costs["curtailment_cost"],
                    "total_cost": costs["total_cost"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[FNO] Done. Saved to: {save_path}")

    def run_sa(
        self,
        max_iter=150,
        T_start=50000.0,
        cooling_rate=0.97,
        reheat_interval=50,
        stall_limit=60,
        lr_warmup_iters=15,
    ):
        save_path, summary_path = self._paths("sa")
        self._init_save(save_path)
        self._init_save(summary_path)
        for i, day in enumerate(self.days):
            print(f"\n{'=' * 60}\n[SA] Day {day}\n{'=' * 60}")
            model_obj = self._build_day(i)
            demand_bus = self.demand_bus_list[i]
            reserve = self.reserve_list[i]
            wind = self.wind_list[i]
            solar = self.solar_list[i]
            N, T = self.params.num_buses, self.params.num_steps
            demand_total = [sum(demand_bus[n][t] for n in range(N)) for t in range(T)]
            solver = SimulatedAnnealingSolver(
                max_iter=max_iter,
                T_start=T_start,
                cooling_rate=cooling_rate,
                reheat_interval=reheat_interval,
                stall_limit=stall_limit,
                lr_warmup_iters=lr_warmup_iters,
                time_limit=self.time_limit,
            )
            t0 = time.time()
            sol = solver.solve(model_obj, demand_bus, demand_total, reserve, wind=wind, solar=solar)
            total_time = time.time() - t0
            summary = {
                "day": day,
                "max_iter": max_iter,
                "T_start": T_start,
                "cooling_rate": cooling_rate,
                "lr_warmup_iters": lr_warmup_iters,
                "n_iters": len(solver.history),
                "n_evals": sum(1 for h in solver.history if h["action"] not in ("skip",)),
                "n_accepted": sum(
                    1 for h in solver.history if h["action"] in ("new_best", "accept")
                ),
                "n_improved": sum(1 for h in solver.history if h["action"] == "new_best"),
                "total_time": round(total_time, 1),
            }
            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, total_time)
            costs = model_obj.cost_breakdown()
            self._print_line_utilization(model_obj)
            self._save_day(day, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "fuel_cost": costs["fuel_cost"],
                    "startup_cost": costs["startup_cost"],
                    "shutdown_cost": costs["shutdown_cost"],
                    "curtailment_cost": costs["curtailment_cost"],
                    "total_cost": costs["total_cost"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[SA] Done. Saved to: {save_path}")

    def run_lns(self, max_iter=50, destroy_ratio=0.08, lr_warmup_iters=25):
        save_path, summary_path = self._paths("lns")
        self._init_save(save_path)
        self._init_save(summary_path)
        for i, day in enumerate(self.days):
            print(f"\n{'=' * 60}\n[LNS] Day {day}\n{'=' * 60}")
            model_obj = self._build_day(i)
            demand_bus = self.demand_bus_list[i]
            reserve = self.reserve_list[i]
            wind = self.wind_list[i]
            solar = self.solar_list[i]
            solver = LargeNeighborhoodSearchSolver(
                max_iter=max_iter,
                destroy_ratio=destroy_ratio,
                lr_warmup_iters=lr_warmup_iters,
                time_limit=self.time_limit,
            )
            t0 = time.time()
            sol = solver.solve(model_obj, demand_bus, reserve, wind, solar)
            total_time = time.time() - t0
            summary = {
                "day": day,
                "max_iter": max_iter,
                "destroy_ratio": destroy_ratio,
                "n_iters": len(solver.history) - 1,
                "total_time": round(total_time, 1),
            }
            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, total_time)
            costs = model_obj.cost_breakdown()
            self._save_day(day, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "fuel_cost": costs["fuel_cost"],
                    "startup_cost": costs["startup_cost"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[LNS] Done. Saved to: {save_path}")

    def run_lr(self, max_iter=30, step_init=0.5, step_decay=0.95, recovery_interval=3):
        save_path, summary_path = self._paths("lr")
        self._init_save(save_path)
        self._init_save(summary_path)
        for i, day in enumerate(self.days):
            print(f"\n{'=' * 60}\n[LR] Day {day}\n{'=' * 60}")
            model_obj = self._build_day(i)
            demand_bus = self.demand_bus_list[i]
            reserve = self.reserve_list[i]
            wind = self.wind_list[i]
            solar = self.solar_list[i]
            solver = LagrangianRelaxationSolver(
                max_iter=max_iter,
                step_init=step_init,
                step_decay=step_decay,
                recovery_interval=recovery_interval,
                time_limit=self.time_limit,
            )
            t0 = time.time()
            sol = solver.solve(model_obj, demand_bus, reserve, wind, solar)
            total_time = time.time() - t0
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
                summary.update({"status": "infeasible", "objective": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, total_time)
            costs = model_obj.cost_breakdown()
            self._print_line_utilization(model_obj)
            self._save_day(day, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "fuel_cost": costs["fuel_cost"],
                    "startup_cost": costs["startup_cost"],
                    "shutdown_cost": costs["shutdown_cost"],
                    "curtailment_cost": costs["curtailment_cost"],
                    "total_cost": costs["total_cost"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[LR] Done. Saved to: {save_path}")

    def run_tabu(
        self, max_iter=100, n_candidates=5, tabu_tenure=10, lr_warmup_iters=25, perturb_ratio=0.15
    ):
        save_path, summary_path = self._paths("tabu")
        self._init_save(save_path)
        self._init_save(summary_path)
        for i, day in enumerate(self.days):
            print(f"\n{'=' * 60}\n[TS] Day {day}\n{'=' * 60}")
            model_obj = self._build_day(i)
            demand_bus = self.demand_bus_list[i]
            reserve = self.reserve_list[i]
            wind = self.wind_list[i]
            solar = self.solar_list[i]
            solver = TabuSearchSolver(
                max_iter=max_iter,
                n_candidates=n_candidates,
                tabu_tenure=tabu_tenure,
                perturb_ratio=perturb_ratio,
                lr_warmup_iters=lr_warmup_iters,
                time_limit=self.time_limit,
            )
            t0 = time.time()
            sol = solver.solve(model_obj, demand_bus, reserve, wind, solar)
            total_time = time.time() - t0
            summary = {
                "day": day,
                "max_iter": max_iter,
                "n_candidates": n_candidates,
                "tabu_tenure": tabu_tenure,
                "n_iters": len(solver.history),
                "total_time": round(total_time, 1),
            }
            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, total_time)
            costs = model_obj.cost_breakdown()
            self._save_day(day, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "fuel_cost": costs["fuel_cost"],
                    "startup_cost": costs["startup_cost"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[TS] Done. Saved to: {save_path}")

    def run_rh(self, horizon_length=24):
        save_path, summary_path = self._paths("rh")
        self._init_save(save_path)
        self._init_save(summary_path)
        for i, day in enumerate(self.days):
            print(f"\n{'=' * 60}\n[RH] Day {day}\n{'=' * 60}")
            model_obj = self._build_day(i)
            demand_bus = self.demand_bus_list[i]
            reserve = self.reserve_list[i]
            wind = self.wind_list[i]
            solar = self.solar_list[i]
            N, T = self.params.num_buses, self.params.num_steps
            demand_total = [sum(demand_bus[n][t] for n in range(N)) for t in range(T)]
            solver = RollingHorizonSolver(horizon_length=horizon_length, time_limit=self.time_limit)
            t0 = time.time()
            sol = solver.solve(model_obj, demand_bus, demand_total, reserve, wind=wind, solar=solar)
            total_time = time.time() - t0
            summary = {
                "day": day,
                "horizon_length": horizon_length,
                "n_windows": len(solver.history),
                "total_time": round(total_time, 1),
            }
            if sol is None:
                print("  Infeasible")
                summary.update({"status": "infeasible", "objective": None})
                self._save_summary(summary, summary_path)
                continue
            self._print_solution(sol, total_time)
            costs = model_obj.cost_breakdown()
            self._print_line_utilization(model_obj)
            self._save_day(day, sol, save_path)
            summary.update(
                {
                    "status": "feasible",
                    "objective": sol["objective"],
                    "fuel_cost": costs["fuel_cost"],
                    "startup_cost": costs["startup_cost"],
                    "shutdown_cost": costs["shutdown_cost"],
                    "curtailment_cost": costs["curtailment_cost"],
                    "total_cost": costs["total_cost"],
                }
            )
            self._save_summary(summary, summary_path)
        print(f"\n[RH] Done. Saved to: {save_path}")


def main(argv=None):
    """Run the command-line entry point for UC experiments."""
    from runners.cli import run_cli

    return run_cli("uc", argv)


if __name__ == "__main__":
    main()
