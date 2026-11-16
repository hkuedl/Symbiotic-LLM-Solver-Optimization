"""Uncertainty-aware LLM sampling and solver-guided refinement."""

import asyncio
import json
import math
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import gurobipy as gp
import numpy as np
import pyomo.environ as pyo

from common.llm_client import create_client
from common.llm_requests import ChatRequests, logprob_content
from common.llm_uc import parse_u_strings, postprocess, print_llm_usage, robust_json_parse
from common.solver_config import (
    DEFAULT_TIME_LIMIT,
    configure_gurobi,
    create_solver,
    solve_model,
    validate_time_limit,
)
from models.uc.model_uc import UnitCommitmentModel
from ours.uc.uncertainty_uc import UncertaintyQuantifier


class OursSolver:
    """OursSolver: uncertainty-aware llm sampling and solver-guided refinement."""

    def __init__(
        self,
        api_key=None,
        base_url=None,
        model="gpt-4o",
        n_samples=5,
        strategy="epistemic_threshold",
        strategy_params=None,
        max_refine_iter=5,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.time_limit = validate_time_limit(time_limit)
        self.client = create_client(api_key, base_url, asynchronous=True)
        self.requests = ChatRequests(self.client)
        self._temp_dir = TemporaryDirectory(prefix="llm_solver_")
        self.model = model
        self.n_samples = n_samples
        self.strategy = strategy
        self.strategy_params = strategy_params or {"tau_I": 0.1}
        self.max_refine_iter = max_refine_iter
        self.solve_time = 0
        self.llm_time = 0
        self.token_usage = {"prompt": 0, "completion": 0}
        self.uq = None
        self.history = []
        self.iter_timings = []

    def solve(self, model_obj, demand_bus, reserve, wind=None, solar=None):
        """Run the method and return the extracted solution, or None on failure."""
        m = model_obj.model
        p = model_obj.p
        if m is None:
            raise RuntimeError("Model not built yet.")
        total_start = time.time()
        total_vars = p.num_generators * p.num_steps
        self.uq = UncertaintyQuantifier(p.num_generators, p.num_steps)
        self.history = []
        self.iter_timings = []
        self.token_usage = {"prompt": 0, "completion": 0}

        G, T = p.num_generators, p.num_steps
        demand_total = [sum(demand_bus[n][t] for n in range(p.num_buses)) for t in range(T)]
        wind_total = None
        solar_total = None
        if wind:
            wind_total = [sum(wind[b][t] for b in wind.keys()) for t in range(T)]
        if solar:
            solar_total = [sum(solar[b][t] for b in solar.keys()) for t in range(T)]
        print(f"[UQ] Phase 1: LLM Sampling ({self.n_samples} samples)...")
        t0 = time.time()
        self._mc_sampling(p, demand_total, reserve, wind_total, solar_total)
        self.llm_time = time.time() - t0
        if self.uq.N == 0:
            print("[UQ] No valid samples. Aborting.")
            return None
        print(f"[UQ] Sampling total: {self.llm_time:.1f}s, valid={self.uq.N}/{self.n_samples}")
        self.uq.summary()

        fixes = self.uq.select_variables_to_fix(strategy=self.strategy, **self.strategy_params)
        fixes = postprocess(fixes, p)
        fixed_keys = set(fixes.keys())
        free_keys = set()
        for g in range(G):
            for t in range(T):
                if (g, t) not in fixed_keys:
                    free_keys.add((g, t))
        print(f"[UQ] fixed={len(fixed_keys)}/{total_vars}, free={len(free_keys)}")
        current_fixes = self._postprocess_commitments(dict(fixes), fixed_keys, p)
        best_fixes = dict(current_fixes)
        best_obj = float("inf")
        best_solution = None
        last_solution = None
        prev_snapshot = None
        infeasible_count = 0
        messages = [
            {"role": "system", "content": self._build_system_prompt(p)},
            {
                "role": "user",
                "content": self._build_user_prompt(demand_total, reserve, wind_total, solar_total),
            },
        ]
        for iteration in range(1, self.max_refine_iter + 1):
            print(f"\n--- Iteration {iteration}/{self.max_refine_iter} ---")
            fixed_on = sum(1 for k in fixed_keys if current_fixes.get(k) == 1)
            fixed_off = sum(1 for k in fixed_keys if current_fixes.get(k) == 0)
            free_count = total_vars - len(fixed_keys)
            print(
                f"  Fixed: {len(fixed_keys)}/{total_vars} "
                f"(ON={fixed_on}, OFF={fixed_off}), "
                f"Free: {free_count} ({free_count / total_vars * 100:.1f}%)"
            )
            iter_timing = {"iter": iteration}
            solve_t0 = time.time()
            sch_obj, milp_m, results, status = self._solve_partial(
                current_fixes,
                fixed_keys,
                p,
                demand_bus,
                reserve,
                wind,
                solar,
                warm_start_sol=last_solution,
            )
            iter_timing["solve_time"] = time.time() - solve_t0
            iter_timing["status"] = status
            if status not in ("optimal", "feasible", "locallyoptimal", "infeasible"):
                self.history.append({"iter": iteration, "status": status})
                iter_timing["llm_refine_time"] = 0
                iter_timing["n_changed"] = 0
                self.iter_timings.append(iter_timing)
                break
            if status == "infeasible":
                self.history.append({"iter": iteration, "status": "infeasible"})
                infeasible_count += 1
                print(f"  INFEASIBLE (count={infeasible_count})")

                try:
                    import gurobipy as gp

                    diag_model = UnitCommitmentModel(p)
                    diag_m = diag_model.build(demand_bus, reserve, wind, solar)
                    for g, t in fixed_keys:
                        diag_m.u[g, t].fix(current_fixes[(g, t)])
                    diag_path = str(Path(self._temp_dir.name) / "_uc_diag.lp")
                    diag_m.write(diag_path, io_options={"symbolic_solver_labels": True})

                    with gp.Env(empty=True) as env:
                        env.setParam("OutputFlag", 0)
                        env.start()
                        gm = gp.read(diag_path, env=env)
                        configure_gurobi(gm, self.time_limit)
                        gm.optimize()
                        if gm.status == gp.GRB.INFEASIBLE:
                            gm.computeIIS()
                            iis_constrs = [c.ConstrName for c in gm.getConstrs() if c.IISConstr]
                            iis_vars = [v.VarName for v in gm.getVars() if v.IISLB or v.IISUB]
                            print(
                                f"  [IIS] {len(iis_constrs)} constraints, "
                                f"{len(iis_vars)} variable bounds"
                            )
                            for c in iis_constrs[:15]:
                                print(f"    Constr: {c}")
                            for v in iis_vars[:10]:
                                print(f"    Var: {v}")
                except Exception as e:
                    print(f"  [IIS] Failed: {str(e)[:100]}")

                if infeasible_count >= 2 and best_fixes is not None:
                    print("  Reverting to best known feasible solution")
                    current_fixes = dict(best_fixes)
                    infeasible_count = 0
                    iter_timing["llm_refine_time"] = 0
                    iter_timing["n_changed"] = 0
                    self.iter_timings.append(iter_timing)
                    continue
                feedback = self._infeasibility_feedback(
                    milp_m,
                    current_fixes,
                    fixed_keys,
                    p,
                    demand_total,
                    reserve,
                    wind_total,
                    solar_total,
                    prev_snapshot,
                )
                fb_type = "infeasible"
            else:
                infeasible_count = 0
                solution = self._extract_solution(sch_obj)
                last_solution = solution
                obj = solution["objective"]
                self.history.append({"iter": iteration, "status": "feasible", "obj": obj})
                iter_timing["objective"] = obj
                total_gen = sum(solution["total_gen"])
                print(f"  obj=${obj:,.0f}, gen={total_gen:,.0f}MWh")
                if obj < best_obj:
                    best_obj = obj
                    best_solution = solution
                    best_fixes = dict(current_fixes)
                    print("  ★ New best!")
                elif obj > best_obj * 1.005:
                    print("  Worsened, reverting to best")
                    current_fixes = dict(best_fixes)
                    iter_timing["llm_refine_time"] = 0
                    iter_timing["n_changed"] = 0
                    self.iter_timings.append(iter_timing)
                    continue
                dual_info = self._extract_dual_info(
                    milp_m,
                    current_fixes,
                    fixed_keys,
                    free_keys,
                    p,
                    demand_bus,
                    reserve,
                    wind,
                    solar,
                )
                if dual_info is None:
                    iter_timing["llm_refine_time"] = 0
                    iter_timing["n_changed"] = 0
                    self.iter_timings.append(iter_timing)
                    break

                n_rc = len(dual_info["reduced_costs"])
                n_lp = len(dual_info["lp_values"])
                n_rc_sig = sum(1 for v in dual_info["reduced_costs"].values() if abs(v) > 0.5)
                n_frac = sum(
                    1
                    for k in fixed_keys
                    if abs(dual_info["lp_values"].get(k, current_fixes[k]) - current_fixes[k])
                    > 0.05
                )
                print(
                    f"  [DUAL] RC entries={n_rc}, LP entries={n_lp}, "
                    f"RC signals={n_rc_sig}, fractional={n_frac}, "
                    f"reserve_binding={dual_info['reserve_binding']}"
                )

                n_signals = self._count_improvement_signals(dual_info, current_fixes, fixed_keys)
                if n_signals == 0:
                    print("  Converged.")
                    iter_timing["llm_refine_time"] = 0
                    iter_timing["n_changed"] = 0
                    self.iter_timings.append(iter_timing)
                    break
                feedback = self._optimality_feedback(
                    dual_info, current_fixes, fixed_keys, p, demand_total, solution, reserve
                )
                fb_type = "optimality"
            if iteration == self.max_refine_iter:
                iter_timing["llm_refine_time"] = 0
                iter_timing["n_changed"] = 0
                self.iter_timings.append(iter_timing)
                break
            if (
                len(self.history) >= 2
                and all(h["status"] == "feasible" for h in self.history[-2:])
                and self.history[-1]["obj"] >= self.history[-2]["obj"]
            ):
                print("  Converged (< 0.1%).")
                iter_timing["llm_refine_time"] = 0
                iter_timing["n_changed"] = 0
                self.iter_timings.append(iter_timing)
                break
            llm_t0 = time.time()
            n_changed, prev_snapshot = self._llm_refine(
                messages,
                feedback,
                fb_type,
                iteration,
                current_fixes,
                fixed_keys,
                p,
                demand_total,
                reserve,
            )
            iter_timing["llm_refine_time"] = time.time() - llm_t0
            iter_timing["n_changed"] = n_changed
            self.iter_timings.append(iter_timing)
            if n_changed == 0:
                break

            current_fixes = self._postprocess_commitments(current_fixes, fixed_keys, p)
            fixed_on = sum(1 for k in fixed_keys if current_fixes.get(k) == 1)
            fixed_off = sum(1 for k in fixed_keys if current_fixes.get(k) == 0)
            free_count = total_vars - len(fixed_keys)
            print(
                f"  After refine: Fixed={len(fixed_keys)} "
                f"(ON={fixed_on}, OFF={fixed_off}), "
                f"Free={free_count} ({free_count / total_vars * 100:.1f}%)"
            )
        self.solve_time = time.time() - total_start
        print(f"\n[UQ] Done. Best=${best_obj:,.0f}, time={self.solve_time:.1f}s")
        self._print_history()
        return best_solution

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return asyncio.run(coro)

    def _mc_sampling(self, p, demand_total, reserve, wind_total, solar_total):
        """Collect samples with token probabilities when the endpoint supports them."""
        system_prompt = self._build_system_prompt(p)
        user_prompt = self._build_user_prompt(demand_total, reserve, wind_total, solar_total)

        async def _single_sample(idx):
            response = await self.requests.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            return idx, response, bool(logprob_content(response))

        async def _run_all():
            tasks = [_single_sample(i) for i in range(self.n_samples)]
            return await asyncio.gather(*tasks)

        t0 = time.time()
        results = self._run_async(_run_all())
        wall_time = time.time() - t0
        print(f"  All {self.n_samples} samples completed in {wall_time:.1f}s")

        G, T = p.num_generators, p.num_steps
        target = (G, T)
        total = G * T

        for idx, response, has_lp in sorted(results, key=lambda x: x[0]):
            usage = response.usage
            self.token_usage["prompt"] += usage.prompt_tokens
            self.token_usage["completion"] += usage.completion_tokens

            raw = response.choices[0].message.content

            try:
                parsed = robust_json_parse(raw)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"  Sample {idx + 1}: SKIPPED ({str(e)[:60]})")
                continue

            if "error" in parsed and "u" not in parsed:
                print(
                    f"  Sample {idx + 1}: SKIPPED "
                    f"(model error: {str(parsed.get('reason', parsed['error']))[:80]})"
                )
                continue

            u_arr = parse_u_strings(parsed.get("u", []), G, T)
            u_list = u_arr.astype(int).tolist()

            logprob_probs = None
            if has_lp:
                logprob_probs = self._extract_logprob_per_variable(logprob_content(response), p)

            self.uq.add_sample(u_list, logprob_probs)

            on_count = int(np.sum(u_arr))
            lp_count = len(logprob_probs) if logprob_probs else 0

            reasoning_tok = 0
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                reasoning_tok = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
            output_tok = usage.completion_tokens - reasoning_tok
            print(
                f"  Sample {idx + 1}: ON={on_count}/{total}, "
                f"logprobs={lp_count}, "
                f"tokens={usage.total_tokens} "
                f"(in={usage.prompt_tokens}, "
                f"out={output_tok}, "
                f"reason={reasoning_tok})"
            )

    def _extract_logprob_per_variable(self, logprobs_content, p):
        if logprobs_content is None:
            return None
        G, T = p.num_generators, p.num_steps
        total_vars = G * T
        full_text = ""
        token_positions = []
        for idx, item in enumerate(logprobs_content):
            start = len(full_text)
            full_text += item.token
            end = len(full_text)
            token_positions.append((start, end, idx))
        u_pattern = re.search(r'"u"\s*:\s*\[', full_text)
        if u_pattern is None:
            return None
        u_array_start = u_pattern.end() - 1
        depth = 0
        u_array_end = u_array_start
        for ci in range(u_array_start, len(full_text)):
            if full_text[ci] == "[":
                depth += 1
            elif full_text[ci] == "]":
                depth -= 1
                if depth == 0:
                    u_array_end = ci + 1
                    break
        string_rows = list(re.finditer(r'"([01]+)"', full_text[u_array_start:u_array_end]))
        if string_rows:
            if len(string_rows) != G or any(len(row.group(1)) != T for row in string_rows):
                return None
            digit_positions = [
                (u_array_start + row.start(1) + offset, digit)
                for row in string_rows
                for offset, digit in enumerate(row.group(1))
            ]
        else:
            digit_pattern = re.compile(r"(?<=[\[,\s])([01])(?=[,\]\s])")
            digit_positions = [
                (match.start(1), match.group(1))
                for match in digit_pattern.finditer(full_text, u_array_start, u_array_end)
            ]
        if len(digit_positions) < total_vars * 0.5:
            return None
        probs = {}
        var_idx = 0
        token_index = 0
        for char_pos, digit_str in digit_positions:
            if var_idx >= total_vars:
                break
            while token_positions[token_index][1] <= char_pos:
                token_index += 1
            item = logprobs_content[token_index]
            offset = char_pos - token_positions[token_index][0]
            opposite = "0" if digit_str == "1" else "1"
            alternative_token = item.token[:offset] + opposite + item.token[offset + 1 :]
            if digit_str == "1":
                p1 = math.exp(item.logprob)
                if item.top_logprobs:
                    for alt in item.top_logprobs:
                        if alt.token == alternative_token:
                            p1 = p1 / (p1 + math.exp(alt.logprob))
                            break
            else:
                p1 = 0.0
                if item.top_logprobs:
                    p_self = math.exp(item.logprob)
                    for alt in item.top_logprobs:
                        if alt.token == alternative_token:
                            p1_raw = math.exp(alt.logprob)
                            p1 = p1_raw / (p_self + p1_raw)
                            break
            g = var_idx // T
            t = var_idx % T
            probs[(g, t)] = float(np.clip(p1, 1e-10, 1 - 1e-10))
            var_idx += 1
        return probs if probs else None

    def _solve_partial(
        self, fixes, fixed_keys, p, demand_bus, reserve, wind, solar, warm_start_sol=None
    ):
        model_obj = UnitCommitmentModel(p)
        m = model_obj.build(demand_bus, reserve, wind, solar)

        for g, t in fixed_keys:
            m.u[g, t].fix(fixes[(g, t)])

        if warm_start_sol is not None:
            self._set_warm_start(m, warm_start_sol, p)

        num_vars = 0
        num_int_vars = 0
        num_constraints = 0

        for v in m.component_data_objects(pyo.Var):
            if not v.is_fixed():
                num_vars += 1
                if v.is_binary() or v.is_integer():
                    num_int_vars += 1

        for c in m.component_data_objects(pyo.Constraint, active=True):
            num_constraints += 1

        print(
            f"  [MODEL] total_vars={num_vars}, "
            f"integer_vars={num_int_vars}, "
            f"constraints={num_constraints}"
        )

        solver = create_solver(self.time_limit)

        t0 = time.time()
        results = solve_model(solver, m, tee=False, warmstart=(warm_start_sol is not None))
        solve_time = time.time() - t0

        status = str(results.solver.termination_condition).lower()
        has_solution = len(results.solution) > 0
        if status not in ("optimal", "feasible", "locallyoptimal") and has_solution:
            status = "feasible"

        if hasattr(results.solver, "statistics"):
            stats = results.solver.statistics
            print(
                f"  [SOLVE] status={status}, time={solve_time:.1f}s, "
                f"nodes={getattr(stats, 'branch_and_bound.number_of_created_subproblems', 'N/A')}"
            )
        else:
            print(
                f"  [SOLVE] status={status}, time={solve_time:.1f}s"
                + (" (warm)" if warm_start_sol else "")
            )

        try:
            lp_path = str(Path(self._temp_dir.name) / "_partial_uc.lp")
            m.write(lp_path, io_options={"symbolic_solver_labels": True})

            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag", 0)
                env.start()
                gm = gp.read(lp_path, env=env)
                if warm_start_sol is not None:
                    pass
                configure_gurobi(gm, self.time_limit)
                gm.optimize()

                print(
                    f"  [GUROBI] MIP_gap={gm.MIPGap:.4f}, "
                    f"nodes_explored={gm.NodeCount}, "
                    f"simplex_iterations={gm.IterCount}"
                )
        except Exception as e:
            print(f"  [GUROBI] Stats extraction failed: {str(e)[:80]}")

        model_obj.model = m
        return model_obj, m, results, status

    def _set_warm_start(self, m, sol, p):
        for g in range(p.num_generators):
            for t in range(p.num_steps):
                m.u[g, t].set_value(round(sol["u"][g][t]))
                m.v[g, t].set_value(round(sol["v"][g][t]))
                m.w[g, t].set_value(round(sol["w"][g][t]))
                if "P" in sol:
                    m.P[g, t].set_value(sol["P"][g][t])

        if p.has_storage:
            for t in range(p.num_steps):
                if "u_ch" in sol and hasattr(m, "u_ch"):
                    m.u_ch[t].set_value(round(sol["u_ch"][t]))
                if "u_dis" in sol and hasattr(m, "u_dis"):
                    m.u_dis[t].set_value(round(sol["u_dis"][t]))
                if "P_ch" in sol:
                    m.P_ch[t].set_value(sol["P_ch"][t])
                if "P_dis" in sol:
                    m.P_dis[t].set_value(sol["P_dis"][t])
                if "SoC" in sol:
                    m.SoC[t].set_value(sol["SoC"][t])

    def _extract_solution(self, model_obj):
        sol = model_obj.extract_results()
        if hasattr(model_obj, "cost_breakdown"):
            sol["cost_breakdown"] = model_obj.cost_breakdown()
        return sol

    def _extract_dual_info(
        self, milp_m, fixes, fixed_keys, free_keys, p, demand_bus, reserve, wind, solar
    ):
        """Solve the original diagnostic relaxation and collect economic signals."""
        all_u = {}
        for g in range(p.num_generators):
            for t in range(p.num_steps):
                all_u[(g, t)] = int(round(pyo.value(milp_m.u[g, t])))

        model_lp = UnitCommitmentModel(p)
        m_lp = model_lp.build(demand_bus, reserve, wind, solar)

        for g, t in free_keys:
            m_lp.u[g, t].fix(all_u[(g, t)])

        for g, t in fixed_keys:
            m_lp.u[g, t].domain = pyo.NonNegativeReals
            m_lp.u[g, t].setub(1.0)
            m_lp.u[g, t].setlb(0.0)
            m_lp.u[g, t].value = fixes[(g, t)]

        for g in range(p.num_generators):
            for t in range(p.num_steps):
                m_lp.v[g, t].domain = pyo.NonNegativeReals
                m_lp.v[g, t].setub(1.0)
                m_lp.w[g, t].domain = pyo.NonNegativeReals
                m_lp.w[g, t].setub(1.0)

        if p.has_storage:
            for t in range(p.num_steps):
                m_lp.u_ch[t].domain = pyo.NonNegativeReals
                m_lp.u_ch[t].setub(1.0)
                m_lp.u_dis[t].domain = pyo.NonNegativeReals
                m_lp.u_dis[t].setub(1.0)

        m_lp.write(
            str(Path(self._temp_dir.name) / "_dual_lp.lp"),
            io_options={"symbolic_solver_labels": True},
        )

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            gm = gp.read(str(Path(self._temp_dir.name) / "_dual_lp.lp"), env=env)

            for v in gm.getVars():
                if v.VType != "C":
                    v.VType = "C"
            gm.update()
            configure_gurobi(gm, self.time_limit)
            gm.optimize()

            if gm.status != gp.GRB.OPTIMAL:
                return None

            info = {
                "lp_obj": gm.objVal,
                "reduced_costs": {},
                "lp_values": {},
                "reserve_binding": False,
            }

            for v in gm.getVars():
                name = v.VarName

                if name.startswith("u(") or name.startswith("u["):
                    nums = re.findall(r"\d+", name)
                    if len(nums) >= 2:
                        g, t = int(nums[0]), int(nums[1])
                        if (g, t) in fixed_keys:
                            info["reduced_costs"][(g, t)] = v.RC
                            info["lp_values"][(g, t)] = v.X

            for c in gm.getConstrs():
                if "ReserveReq" in c.ConstrName and abs(c.Pi) > 1e-6:
                    info["reserve_binding"] = True
                    break

        return info

    def _count_improvement_signals(self, dual_info, fixes, fixed_keys):
        rc = dual_info["reduced_costs"]
        lp_vals = dual_info["lp_values"]
        n_frac = sum(1 for k in fixed_keys if abs(lp_vals.get(k, fixes[k]) - fixes[k]) > 0.05)
        n_rc = sum(1 for v in rc.values() if abs(v) > 0.5)
        return n_frac + n_rc

    def _infeasibility_feedback(
        self,
        milp_m,
        current_fixes,
        fixed_keys,
        p,
        demand_total,
        reserve,
        wind_total,
        solar_total,
        prev_snapshot=None,
    ):
        fb = ["INFEASIBLE. Adjust commitment decisions to restore feasibility."]

        if prev_snapshot is not None:
            diff = []
            for key in fixed_keys:
                if current_fixes[key] != prev_snapshot.get(key, current_fixes[key]):
                    diff.append((key, prev_snapshot[key], current_fixes[key]))
            if diff:
                fb.append(f"\nLast changes ({len(diff)} vars) — revert some:")
                for (g, t), old, new in diff[:15]:
                    fb.append(f"  G{g}_T{t}: {old} -> {new} (revert to {old})")

        init_violations = []
        for g in range(p.num_generators):
            status_h = p.initial_status_steps[g]
            if status_h > 0:
                remaining = max(0, p.min_up_time[g] - status_h)
                for t in range(min(remaining, p.num_steps)):
                    if (g, t) in fixed_keys and current_fixes[(g, t)] == 0:
                        init_violations.append((g, t, 1))
            elif status_h < 0:
                remaining = max(0, p.min_down_time[g] - abs(status_h))
                for t in range(min(remaining, p.num_steps)):
                    if (g, t) in fixed_keys and current_fixes[(g, t)] == 1:
                        init_violations.append((g, t, 0))
        if init_violations:
            fb.append(f"\n[MUST FIX] Initial state violations ({len(init_violations)}):")
            for g, t, correct in init_violations:
                fb.append(f"  G{g}_T{t} -> must be {correct}")

        mintime_issues = []
        for g in range(p.num_generators):
            for t in range(1, p.num_steps):
                if (g, t) not in fixed_keys or (g, t - 1) not in fixed_keys:
                    continue
                if current_fixes[(g, t)] == 1 and current_fixes[(g, t - 1)] == 0:
                    on_len = 1
                    for tt in range(t + 1, p.num_steps):
                        if (g, tt) in fixed_keys and current_fixes[(g, tt)] == 1:
                            on_len += 1
                        else:
                            break
                    t_end = t + on_len
                    if (
                        on_len < p.min_up_time[g]
                        and t_end < p.num_steps
                        and (g, t_end) in fixed_keys
                        and current_fixes[(g, t_end)] == 0
                    ):
                        mintime_issues.append(
                            (g, t, t + on_len - 1, "min_up", p.min_up_time[g], on_len)
                        )
                if current_fixes[(g, t)] == 0 and current_fixes[(g, t - 1)] == 1:
                    off_len = 1
                    for tt in range(t + 1, p.num_steps):
                        if (g, tt) in fixed_keys and current_fixes[(g, tt)] == 0:
                            off_len += 1
                        else:
                            break
                    t_end = t + off_len
                    if (
                        off_len < p.min_down_time[g]
                        and t_end < p.num_steps
                        and (g, t_end) in fixed_keys
                        and current_fixes[(g, t_end)] == 1
                    ):
                        mintime_issues.append(
                            (g, t, t + off_len - 1, "min_down", p.min_down_time[g], off_len)
                        )

        if mintime_issues:
            fb.append(f"\n[MUST FIX] Min-time violations ({len(mintime_issues)}):")
            for g, ts, te, vtype, req, actual in mintime_issues:
                if vtype == "min_up":
                    fb.append(
                        f"  G{g} steps {ts}-{te}: ON for {actual} < min_up={req} "
                        f"-> extend ON to step {ts + req - 1} or release steps {ts}-{te}"
                    )
                else:
                    fb.append(
                        f"  G{g} steps {ts}-{te}: OFF for {actual} < min_down={req} "
                        f"-> extend OFF to step {ts + req - 1} or release steps {ts}-{te}"
                    )

        capacity_gaps = []
        for t in range(p.num_steps):
            online_cap = sum(
                p.P_max[g]
                for g in range(p.num_generators)
                if (g, t) in fixed_keys and current_fixes[(g, t)] == 1
            )
            free_cap = sum(p.P_max[g] for g in range(p.num_generators) if (g, t) not in fixed_keys)
            re_avail = 0
            if wind_total:
                re_avail += wind_total[t]
            if solar_total:
                re_avail += solar_total[t]
            need = demand_total[t] + reserve[t]
            gap = need - (online_cap + free_cap + re_avail)
            if gap > 0:
                capacity_gaps.append((gap, t, online_cap, free_cap, re_avail, need))

        if capacity_gaps:
            capacity_gaps.sort(reverse=True)
            fb.append(f"\n[CAPACITY SHORT] {len(capacity_gaps)} steps under-capacity:")
            for gap, t, on_cap, free_cap, re_avail, need in capacity_gaps:
                fb.append(
                    f"  Step {t}: gap={gap:.0f}MW "
                    f"(need={need:.0f}, on_cap={on_cap:.0f}, "
                    f"free={free_cap:.0f}, RE={re_avail:.0f})"
                )
                off_gens = [
                    (p.cost_b[g], p.P_max[g], g)
                    for g in range(p.num_generators)
                    if (g, t) in fixed_keys and current_fixes[(g, t)] == 0
                ]
                off_gens.sort()
                cumulative = 0
                for cost, pmax, g in off_gens:
                    fb.append(f"    G{g}_T{t} -> set 1 (cost_b={cost:.1f}, Pmax={pmax:.0f})")
                    cumulative += pmax
                    if cumulative >= gap:
                        break

        pmin_overloads = []
        for t in range(p.num_steps):
            online_Pmin = sum(
                p.P_min[g]
                for g in range(p.num_generators)
                if (g, t) in fixed_keys and current_fixes[(g, t)] == 1
            )
            if online_Pmin > demand_total[t]:
                excess = online_Pmin - demand_total[t]
                pmin_overloads.append((excess, t, online_Pmin))

        if pmin_overloads:
            pmin_overloads.sort(reverse=True)
            fb.append(f"\n[PMIN OVERLOAD] {len(pmin_overloads)} steps:")
            for excess, t, online_Pmin in pmin_overloads:
                fb.append(
                    f"  Step {t}: Pmin_total={online_Pmin:.0f} > "
                    f"demand={demand_total[t]:.0f} (excess={excess:.0f})"
                )
                on_gens = [
                    (p.P_min[g], p.cost_b[g], g)
                    for g in range(p.num_generators)
                    if (g, t) in fixed_keys and current_fixes[(g, t)] == 1
                ]
                on_gens.sort(reverse=True)
                cumulative = 0
                for pmin, cost, g in on_gens:
                    fb.append(f"    G{g}_T{t} -> set 0 (Pmin={pmin:.1f}, cost_b={cost:.1f})")
                    cumulative += pmin
                    if cumulative >= excess:
                        break

        ramp_violations = []
        for g in range(p.num_generators):
            for t in range(1, p.num_steps):
                if (g, t) not in fixed_keys or (g, t - 1) not in fixed_keys:
                    continue
                if current_fixes[(g, t)] == 1 and current_fixes[(g, t - 1)] == 1:
                    pass
                elif current_fixes[(g, t)] == 1 and current_fixes[(g, t - 1)] == 0:
                    if p.P_min[g] > p.startup_ramp[g]:
                        ramp_violations.append((g, t, "startup", p.P_min[g], p.startup_ramp[g]))
                elif current_fixes[(g, t)] == 0 and current_fixes[(g, t - 1)] == 1:
                    if p.P_min[g] > p.ramp_down[g] + p.shutdown_ramp[g]:
                        ramp_violations.append(
                            (g, t - 1, "shutdown", p.P_min[g], p.ramp_down[g] + p.shutdown_ramp[g])
                        )

        if ramp_violations:
            fb.append(f"\n[RAMP VIOLATION] {len(ramp_violations)} transitions:")
            for g, t, vtype, pmin, limit in ramp_violations:
                fb.append(
                    f"  G{g}_T{t}: {vtype} ramp infeasible "
                    f"(Pmin={pmin:.0f} > limit={limit:.0f}) -> release G{g}_T{t}"
                )

        priority_order = []
        if init_violations:
            priority_order.append("initial-state violations")
        if mintime_issues:
            priority_order.append("min-time violations")
        if ramp_violations:
            priority_order.append("ramp violations")
        if capacity_gaps:
            priority_order.append("capacity shortages")
        if pmin_overloads:
            priority_order.append("Pmin overloads")

        if priority_order:
            fb.append(f"\nFix priority: {' > '.join(priority_order)}")
        fb.append("Prefer releasing uncertain variables over hard-setting when unsure.")

        return "\n".join(fb)

    def _optimality_feedback(
        self, dual_info, fixes, fixed_keys, p, demand_total, solution, reserve
    ):
        fb = []

        cb = solution.get("cost_breakdown", {})
        if cb:
            fb.append(
                f"Objective=${solution['objective']:,.0f} "
                f"(fuel={cb.get('fuel_cost', 0):,.0f}, "
                f"startup={cb.get('startup_cost', 0):,.0f})"
            )
        else:
            fb.append(f"Objective: ${solution['objective']:,.0f}")

        fb.append(
            f"LP lower bound: {dual_info['lp_obj']:,.0f} "
            f"(gap: {solution['objective'] - dual_info['lp_obj']:,.0f})"
        )

        if dual_info["reserve_binding"]:
            fb.append("Reserve is tight — do not turn off large generators.")

        rc = dual_info["reduced_costs"]
        lp_vals = dual_info["lp_values"]

        fractional = []
        for g, t in fixed_keys:
            fix_val = fixes[(g, t)]
            lp_val = lp_vals.get((g, t), fix_val)
            rc_val = rc.get((g, t), 0)
            deviation = abs(lp_val - fix_val)
            if deviation > 0.05:
                est_cost = p.cost_b[g] * p.P_max[g] * p.dt
                fractional.append((deviation, rc_val, g, t, fix_val, lp_val, est_cost))
        fractional.sort(reverse=True)

        if fractional:
            should_off = [
                (d, r, g, t, est) for d, r, g, t, fv, lv, est in fractional if fv == 1 and lv < 0.5
            ]
            should_on = [
                (d, r, g, t, est) for d, r, g, t, fv, lv, est in fractional if fv == 0 and lv > 0.5
            ]

            if should_off:
                blocks = self._aggregate_blocks(should_off)
                total_saving = sum(b["est_saving"] for b in blocks[:10])
                fb.append(
                    f"\nSET to OFF ({len(should_off)} vars, est. saving ~${total_saving:,.0f}):"
                )
                for b in blocks[:10]:
                    fb.append(
                        f"  G{b['g']} steps {b['t_start']}-{b['t_end']}: "
                        f"save ~${b['est_saving']:,.0f}, "
                        f"cost_b={p.cost_b[b['g']]:.1f} → set 0"
                    )

            if should_on:
                blocks = self._aggregate_blocks(should_on)
                fb.append(f"\nSET to ON ({len(should_on)} vars):")
                for b in blocks[:10]:
                    fb.append(
                        f"  G{b['g']} steps {b['t_start']}-{b['t_end']}: "
                        f"cost_b={p.cost_b[b['g']]:.1f} → set 1"
                    )

            release_candidates = []
            for g, t in fixed_keys:
                fv = fixes[(g, t)]
                lv = lp_vals.get((g, t), fv)
                rc_val = rc.get((g, t), 0)
                deviation = abs(lv - fv)
                if 0.01 < deviation <= 0.05 or (deviation <= 0.05 and abs(rc_val) > 1.0):
                    release_candidates.append({"g": g, "t": t, "rc": rc_val})
            release_candidates.sort(key=lambda x: -abs(x["rc"]))

            if release_candidates:
                release_blocks = self._aggregate_release_blocks(release_candidates)
                fb.append(f"\nRELEASE ({len(release_candidates)} borderline vars):")
                for b in release_blocks[:10]:
                    fb.append(f"  G{b['g']} steps {b['t_start']}-{b['t_end']} → release")
        else:
            should_off_rc = sorted(
                [
                    (rc.get(k, 0), k[0], k[1])
                    for k in fixed_keys
                    if fixes[k] == 1 and rc.get(k, 0) > 0.5
                ],
                reverse=True,
            )
            should_on_rc = sorted(
                [
                    (abs(rc.get(k, 0)), k[0], k[1])
                    for k in fixed_keys
                    if fixes[k] == 0 and rc.get(k, 0) < -0.5
                ],
                reverse=True,
            )
            if should_off_rc:
                fb.append("\nBy RC — SET to OFF:")
                for rv, g, t in should_off_rc[:8]:
                    fb.append(f"  G{g}_T{t}: RC={rv:.1f} → set 0")
            if should_on_rc:
                fb.append("\nBy RC — SET to ON:")
                for rv, g, t in should_on_rc[:8]:
                    fb.append(f"  G{g}_T{t}: RC={rv:.1f} → set 1")

            mid_rc = sorted(
                [
                    (abs(rc.get(k, 0)), k[0], k[1])
                    for k in fixed_keys
                    if 0.1 < abs(rc.get(k, 0)) <= 0.5
                ],
                reverse=True,
            )
            if mid_rc:
                fb.append(f"\nRELEASE ({len(mid_rc)} mid-range RC vars):")
                for rv, g, t in mid_rc[:10]:
                    fb.append(f"  G{g}_T{t} → release")

            if not should_off_rc and not should_on_rc and not mid_rc:
                fb.append("\nNo improvement signals found.")

        structural = self._structural_feedback(solution, p, demand_total, reserve)
        if structural:
            fb.append(f"\n{structural}")

        fb.append(f"\nDemand range: {min(demand_total):.0f}-{max(demand_total):.0f} MW.")

        return "\n".join(fb)

    def _structural_feedback(self, solution, p, demand_total, reserve):
        gens = []
        for g in range(p.num_generators):
            on_steps = sum(1 for t in range(p.num_steps) if solution["u"][g][t] > 0.5)
            if on_steps == 0:
                continue

            total_output = sum(solution["P"][g][t] * p.dt for t in range(p.num_steps))
            utilization = (
                total_output / (on_steps * p.dt * p.P_max[g])
                if on_steps > 0 and p.P_max[g] > 0
                else 0
            )

            startup_count = sum(1 for t in range(p.num_steps) if solution["v"][g][t] > 0.5)

            cost_per_mwh = (
                total_output * p.cost_b[g]
                + startup_count * p.cost_startup[g]
                + p.cost_c[g] * on_steps * p.dt
            ) / max(total_output, 1)

            gens.append(
                {
                    "g": g,
                    "on_steps": on_steps,
                    "total_output": total_output,
                    "utilization": utilization,
                    "cost_per_mwh": cost_per_mwh,
                    "startup_count": startup_count,
                }
            )

        gens.sort(key=lambda x: x["cost_per_mwh"])

        fb = ["GENERATOR ANALYSIS:"]
        for info in gens[:8]:
            fb.append(
                f"  G{info['g']}: {info['total_output']:.0f}MWh, "
                f"${info['cost_per_mwh']:.1f}/MWh, "
                f"util={info['utilization']:.0%}, "
                f"on={info['on_steps']}steps, "
                f"startups={info['startup_count']}"
            )

        if len(gens) >= 2:
            cheapest = gens[0]
            costliest = gens[-1]
            if costliest["cost_per_mwh"] > cheapest["cost_per_mwh"] * 1.5:
                fb.append(
                    f"\n  G{costliest['g']} costs "
                    f"${costliest['cost_per_mwh']:.1f}/MWh vs "
                    f"G{cheapest['g']} ${cheapest['cost_per_mwh']:.1f}/MWh. "
                    f"Reduce G{costliest['g']} runtime."
                )

        for info in gens:
            if info["startup_count"] < 2:
                continue
            g = info["g"]
            on_blocks = []
            t = 0
            while t < p.num_steps:
                if solution["u"][g][t] > 0.5:
                    t_start = t
                    while t < p.num_steps and solution["u"][g][t] > 0.5:
                        t += 1
                    on_blocks.append((t_start, t - 1))
                else:
                    t += 1
            for i in range(len(on_blocks) - 1):
                gap = on_blocks[i + 1][0] - on_blocks[i][1] - 1
                if gap <= p.min_down_time[g] + 2:
                    fb.append(
                        f"\n  G{g}: gap {gap} steps between "
                        f"blocks {on_blocks[i]} and {on_blocks[i + 1]}. "
                        f"Merge to save ${p.cost_startup[g]:,.0f}."
                    )
                    break

        low_util = [i for i in gens if i["utilization"] < 0.3 and i["on_steps"] > 10]
        if low_util:
            fb.append("\n  Low utilization:")
            for info in low_util[:3]:
                fb.append(
                    f"    G{info['g']}: util={info['utilization']:.0%}, "
                    f"on={info['on_steps']}steps → consider turning off."
                )

        return "\n".join(fb)

    def _aggregate_blocks(self, items):
        by_gen = {}
        for dev, rc_v, g, t, est in items:
            by_gen.setdefault(g, []).append((t, dev, rc_v, est))
        blocks = []
        for g, entries in by_gen.items():
            entries.sort()
            i = 0
            while i < len(entries):
                t_s = entries[i][0]
                total_dev = entries[i][1]
                total_est = entries[i][3]
                j = i + 1
                while j < len(entries) and entries[j][0] == entries[j - 1][0] + 1:
                    total_dev += entries[j][1]
                    total_est += entries[j][3]
                    j += 1
                n_steps = j - i
                blocks.append(
                    {
                        "g": g,
                        "t_start": t_s,
                        "t_end": entries[j - 1][0],
                        "n_steps": n_steps,
                        "total_dev": total_dev,
                        "est_saving": total_est,
                    }
                )
                i = j
        blocks.sort(key=lambda x: -x["est_saving"])
        return blocks

    def _aggregate_release_blocks(self, items):
        by_gen = {}
        for c in items:
            by_gen.setdefault(c["g"], []).append(c["t"])
        blocks = []
        for g, times in by_gen.items():
            times.sort()
            i = 0
            while i < len(times):
                t_s = times[i]
                j = i + 1
                while j < len(times) and times[j] == times[j - 1] + 1:
                    j += 1
                blocks.append(
                    {
                        "g": g,
                        "t_start": t_s,
                        "t_end": times[j - 1],
                        "n_steps": j - i,
                    }
                )
                i = j
        blocks.sort(key=lambda x: -x["n_steps"])
        return blocks

    def _llm_refine(
        self,
        messages,
        feedback,
        fb_type,
        iteration,
        current_fixes,
        fixed_keys,
        p,
        demand_total,
        reserve,
    ):
        action = "fix infeasibility" if fb_type == "infeasible" else "reduce cost"
        prompt_msg = (
            f"=== SOLVER FEEDBACK (Iter {iteration}, {fb_type}) ===\n\n"
            f"{feedback}\n\n"
            f"Change some fixed variables to {action}.\n"
            f'Output: JSON with "changes" list. Each entry:\n'
            f'{{"g": int, "t_start": int, "t_end": int, '
            f'"action": "set"|"release", "value": 0 or 1}}\n'
            f'"set": fix the variable to value 0 or 1.\n'
            f'"release": remove from fixed set, let solver decide optimally. value is ignored for release.\n'
            f"Use release for uncertain variables. Empty list if none.\n"
            f"Output only the JSON object, no other text."
        )
        messages.append({"role": "user", "content": prompt_msg})

        async def _call():
            return await self.requests.acreate(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )

        t0 = time.time()
        response = self._run_async(_call())
        llm_time = time.time() - t0
        usage = response.usage
        self.token_usage["prompt"] += usage.prompt_tokens
        self.token_usage["completion"] += usage.completion_tokens
        print_llm_usage(usage, llm_time)
        raw_resp = response.choices[0].message.content
        messages.append({"role": "assistant", "content": raw_resp})
        try:
            parsed = robust_json_parse(raw_resp)
            if isinstance(parsed, list):
                changes = parsed
            elif isinstance(parsed, dict):
                changes = parsed.get("changes", [])
            else:
                changes = []
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  LLM refine parse failed: {str(e)[:60]}")
            changes = []

        changes = self._validate_changes(
            changes, current_fixes, fixed_keys, p, demand_total, reserve
        )
        snapshot = dict(current_fixes)
        n_changed = 0
        n_released = 0
        for c in changes:
            g = c.get("g", -1)
            t_start = max(0, c.get("t_start", -1))
            t_end = min(c.get("t_end", t_start), p.num_steps - 1)
            action_type = c.get("action", "set")
            value = int(c.get("value", 0))
            for t in range(t_start, t_end + 1):
                if (g, t) not in fixed_keys:
                    continue
                if action_type == "release":
                    fixed_keys.discard((g, t))
                    current_fixes.pop((g, t), None)
                    n_released += 1
                elif current_fixes.get((g, t)) != value:
                    current_fixes[(g, t)] = value
                    n_changed += 1
        print(f"  Applied {n_changed} changes, released {n_released} vars")
        return n_changed + n_released, snapshot

    def _validate_changes(self, changes, current_fixes, fixed_keys, p, demand_total, reserve):
        """Validate proposed changes before updating fixed assignments."""
        valid = []
        for c in changes:
            g = c.get("g", -1)
            t_start = max(0, c.get("t_start", -1))
            t_end = min(c.get("t_end", t_start), p.num_steps - 1)
            value = int(c.get("value", 0))
            action_type = c.get("action", "set")

            if g < 0 or g >= p.num_generators:
                continue
            if t_start < 0 or t_end < t_start:
                continue
            if action_type not in ("set", "release"):
                continue

            valid.append(c)

        return valid

    def _postprocess_commitments(self, fixes, fixed_keys, p):
        fixes = dict(fixes)

        for g in range(p.num_generators):
            max_shutdown = p.ramp_down[g] + p.shutdown_ramp[g]
            if p.P_min[g] > max_shutdown:
                for t in range(p.num_steps):
                    if (g, t) in fixed_keys and fixes.get((g, t)) == 0:
                        fixes.pop((g, t), None)
                        fixed_keys.discard((g, t))

        for g in range(p.num_generators):
            status_h = p.initial_status_steps[g]
            if status_h > 0:
                remaining = max(0, p.min_up_time[g] - status_h)
                for t in range(min(remaining, p.num_steps)):
                    if (g, t) in fixed_keys and fixes.get((g, t)) == 0:
                        fixes[(g, t)] = 1
            elif status_h < 0:
                remaining = max(0, p.min_down_time[g] - abs(status_h))
                for t in range(min(remaining, p.num_steps)):
                    if (g, t) in fixed_keys and fixes.get((g, t)) == 1:
                        fixes[(g, t)] = 0

        for g in range(p.num_generators):
            t = 0
            while t < p.num_steps:
                if (g, t) in fixed_keys and fixes.get((g, t)) == 1:
                    t_start = t
                    while t < p.num_steps:
                        if (g, t) in fixed_keys and fixes.get((g, t)) == 1:
                            t += 1
                        elif (g, t) not in fixed_keys:
                            t += 1
                        else:
                            break

                    fixed_on_count = sum(
                        1
                        for tt in range(t_start, t)
                        if (g, tt) in fixed_keys and fixes.get((g, tt)) == 1
                    )
                    block_len = t - t_start

                    if block_len < p.min_up_time[g]:
                        for tt in range(t_start, t):
                            if (g, tt) in fixed_keys and fixes.get((g, tt)) == 1:
                                fixes.pop((g, tt), None)
                                fixed_keys.discard((g, tt))
                else:
                    t += 1

        for g in range(p.num_generators):
            t = 0
            while t < p.num_steps:
                if (g, t) in fixed_keys and fixes.get((g, t)) == 0:
                    t_start = t
                    while t < p.num_steps:
                        if (g, t) in fixed_keys and fixes.get((g, t)) == 0:
                            t += 1
                        elif (g, t) not in fixed_keys:
                            t += 1
                        else:
                            break
                    block_len = t - t_start

                    if block_len < p.min_down_time[g]:
                        for tt in range(t_start, t):
                            if (g, tt) in fixed_keys and fixes.get((g, tt)) == 0:
                                fixes.pop((g, tt), None)
                                fixed_keys.discard((g, tt))
                else:
                    t += 1

        return fixes

    def _print_history(self):
        print(f"\n  {'Iter':>5s} {'Status':>12s} {'Objective':>12s}")
        print(f"  {'-' * 32}")
        for h in self.history:
            if h["status"] == "feasible":
                print(f"  {h['iter']:>5d} {'feasible':>12s} {h['obj']:>12,.0f}")
            else:
                print(f"  {h['iter']:>5d} {'INFEASIBLE':>12s} {'---':>12s}")

    def _build_system_prompt(self, p):
        G = p.num_generators
        T = p.num_steps
        N = p.num_buses

        gen_by_cost = sorted(range(G), key=lambda g: p.cost_b[g])
        n3 = G // 3
        cheap = gen_by_cost[:n3]
        mid = gen_by_cost[n3 : 2 * n3]
        expensive = gen_by_cost[2 * n3 :]

        def group_summary(gens, label):
            total_cap = sum(p.P_max[g] for g in gens)
            total_pmin = sum(p.P_min[g] for g in gens)
            avg_cost = float(np.mean([p.cost_b[g] for g in gens]))
            avg_startup = float(np.mean([p.cost_startup[g] for g in gens]))
            return (
                f"  {label}: {len(gens)} units, "
                f"total_cap={total_cap:.0f}MW, "
                f"total_Pmin={total_pmin:.0f}MW, "
                f"avg_cost={avg_cost:.1f}$/MWh, "
                f"avg_startup=${avg_startup:.0f}"
            )

        group_text = "\n".join(
            [
                group_summary(cheap, "Baseload (cheap)"),
                group_summary(mid, "Mid-merit"),
                group_summary(expensive, "Peaker (expensive)"),
            ]
        )

        gen_info = []
        for g in range(G):
            gen_info.append(
                {
                    "g": g,
                    "bus": p.gen_bus[g],
                    "P_min": round(p.P_min[g], 1),
                    "P_max": round(p.P_max[g], 1),
                    "cost_b": round(p.cost_b[g], 2),
                    "startup_cost": round(p.cost_startup[g], 0),
                    "min_up": p.min_up_time[g],
                    "min_down": p.min_down_time[g],
                }
            )

        init_constrained = []
        for g in range(G):
            status_h = p.initial_status_steps[g]
            if status_h > 0:
                remaining = max(0, p.min_up_time[g] - status_h)
                if remaining > 0:
                    init_constrained.append(
                        f"G{g}(bus={p.gen_bus[g]}): ON, must stay ON for {remaining} more steps"
                    )
            elif status_h < 0:
                remaining = max(0, p.min_down_time[g] - abs(status_h))
                if remaining > 0:
                    init_constrained.append(
                        f"G{g}(bus={p.gen_bus[g]}): OFF, must stay OFF for {remaining} more steps"
                    )

        init_text = (
            "\n".join(init_constrained) if init_constrained else "No binding initial constraints."
        )

        gen_at_bus = {}
        for g in range(G):
            gen_at_bus.setdefault(p.gen_bus[g], []).append(g)

        bus_gen_info = []
        for bus in sorted(gen_at_bus.keys()):
            gens = gen_at_bus[bus]
            cap = sum(p.P_max[g] for g in gens)
            bus_gen_info.append(
                f"Bus {bus}: {', '.join(f'G{g}({p.P_max[g]:.0f}MW)' for g in gens)}, "
                f"total={cap:.0f}MW"
            )

        is_claude = "claude" in self.model.lower()
        if is_claude:
            output_instruction = (
                f"Output: JSON with only one key:\n"
                f'- "u": an array of {G} binary strings, each {T} chars of "0"/"1".\n'
                f'  u[g] = "101100..." means generator g is ON(1)/OFF(0) at each step.\n'
                f'  Example for 3 generators, 4 steps: {{"u": ["1100", "1111", "0011"]}}\n'
                f"Output ONLY a valid JSON object, nothing else. "
                f"Do NOT ask questions. Do NOT explain. Do NOT refuse. "
                f"Use your best judgment for any ambiguity. "
                f"Your entire response must be parseable by json.loads()."
            )
        else:
            output_instruction = (
                f"Output: JSON with only one key:\n"
                f'- "u": an array of {G} binary strings, each {T} chars of "0"/"1".\n'
                f'  u[g] = "101100..." means generator g is ON(1)/OFF(0) at each step.\n'
                f'  Example for 3 generators, 4 steps: {{"u": ["1100", "1111", "0011"]}}\n'
                f"Output ONLY the JSON object. No other text."
            )

        return f"""Task: You are a pre-decision advisor for a power system unit commitment problem. Suggest on/off decisions for generators to minimize total generation cost while satisfying demand, spinning reserve, and transmission network constraints. The MILP solver will optimally determine continuous dispatch levels given your commitment decisions.

    System:
    {G} generators, {N} buses, {p.num_lines} transmission lines (capacity {p.line_capacity[0]:.0f}MW each).
    Time horizon: {T} steps, each {p.dt} hours. Index starts from 0.

    Objective: minimize fuel cost + startup/shutdown costs + curtailment cost.

    Generator classification (by marginal cost):
    {group_text}

    Generator parameters:
    {json.dumps(gen_info, indent=1)}

    Generation by bus (network topology):
    {chr(10).join(bus_gen_info)}

    Initial state constraints:
    {init_text}

    Rules:
    1. Minimum up/down time: each generator must remain ON for at least min_up consecutive steps once started, and OFF for at least min_down consecutive steps once shut down.
    2. System reserve requirement: total online capacity (sum of P_max of committed generators) must exceed total demand plus spinning reserve at every time step.
    3. Minimum generation constraint: total minimum output of committed generators (sum of P_min) must not exceed total demand at any time step, to ensure feasible economic dispatch.
    4. Initial state compliance: generators with binding initial conditions (must_stay_on or must_stay_off) shall maintain their current state for the specified number of remaining steps.
    5. Transmission capacity: power flow on each line is bounded by its thermal limit ({p.line_capacity[0]:.0f}MW). Buses with local demand exceeding local generation capacity depend on imports through adjacent lines.

    Guidelines:
    1. Baseload commitment: generators with low marginal cost should remain committed throughout the horizon to avoid incurring unnecessary startup costs.
    2. Peak shaving: high-cost peaking units should be committed only during periods where net demand approaches system capacity limits.
    3. Startup cost avoidance: toggling a generator on and off incurs startup cost; prefer sustained commitment over intermittent cycling when economically justified.
    4. Network-aware commitment: ensure that buses with significant local demand have sufficient local generation committed, as transmission capacity constraints may prevent adequate power imports from remote generators.
    5. Off-peak decommitment: during low-demand periods, decommit expensive units while maintaining sufficient online capacity to satisfy both demand and reserve requirements.
    6. Temporal coordination: avoid creating commitment blocks shorter than the minimum up/down time, as such schedules are infeasible.

    {output_instruction}"""

    def _build_user_prompt(self, demand_total, reserve, wind_total, solar_total):
        parts = []

        d = demand_total
        peak_t = int(np.argmax(d))
        valley_t = int(np.argmin(d))
        parts.append(
            f"Demand pattern: peak={max(d):.0f}MW at t={peak_t}, "
            f"valley={min(d):.0f}MW at t={valley_t}, "
            f"avg={np.mean(d):.0f}MW"
        )

        parts.append(f"\nDemand (MW) [{len(d)} steps]:")
        parts.append(json.dumps([round(v, 1) for v in d]))

        parts.append("\nReserve requirement (MW):")
        parts.append(json.dumps([round(v, 1) for v in reserve]))

        if wind_total:
            parts.append("\nWind available (MW):")
            parts.append(json.dumps([round(v, 1) for v in wind_total]))

        if solar_total:
            parts.append("\nSolar available (MW):")
            parts.append(json.dumps([round(v, 1) for v in solar_total]))

        if wind_total or solar_total:
            net = np.array(d)
            if wind_total:
                net = net - np.array(wind_total)
            if solar_total:
                net = net - np.array(solar_total)
            parts.append("\nNet demand (demand - renewables) (MW):")
            parts.append(json.dumps([round(v, 1) for v in net.tolist()]))

        return "\n".join(parts)
