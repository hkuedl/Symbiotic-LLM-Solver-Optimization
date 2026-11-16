"""Reference optimization method."""

import json
import time

import numpy as np
import pyomo.environ as pyo

from common.llm_client import create_client
from common.llm_requests import ChatRequests
from common.llm_uc import parse_u_strings, postprocess, print_llm_usage, robust_json_parse
from common.solver_config import DEFAULT_TIME_LIMIT, create_solver, solve_model, validate_time_limit


class LLMSolver:
    """LLMSolver: reference optimization method."""

    def __init__(self, api_key=None, base_url=None, model="gpt-4o", time_limit=DEFAULT_TIME_LIMIT):
        self.time_limit = validate_time_limit(time_limit)
        self.client = create_client(api_key, base_url)
        self.requests = ChatRequests(self.client)
        self.model = model
        self.solve_time = 0
        self.llm_time = 0
        self.token_usage = None

    def solve(self, model_obj, demand_bus, reserve, wind=None, solar=None):
        """Run the method and return the extracted solution, or None on failure."""
        m = model_obj.model
        p = model_obj.p
        if m is None:
            raise RuntimeError("Model not built yet.")

        total_start = time.time()

        G, T = p.num_generators, p.num_steps
        demand_total = [sum(demand_bus[n][t] for n in range(p.num_buses)) for t in range(T)]
        wind_total = None
        solar_total = None
        if wind:
            wind_total = [sum(wind[b][t] for b in wind.keys()) for t in range(T)]
        if solar:
            solar_total = [sum(solar[b][t] for b in solar.keys()) for t in range(T)]

        print(f"[LLM] Calling {self.model}...")
        t0 = time.time()
        u_arr = self._call_llm(p, demand_total, reserve, wind_total, solar_total)
        self.llm_time = time.time() - t0
        print_llm_usage(self.token_usage, self.llm_time)

        if u_arr is None:
            return None

        fixes = {}
        for g in range(G):
            for t in range(T):
                fixes[(g, t)] = int(u_arr[g, t])

        fixes = postprocess(fixes, p)

        for g, t in fixes.keys():
            m.u[g, t].fix(fixes[(g, t)])

        total = G * T
        on_count = sum(fixes.values())
        print(f"[LLM] Fixed all {total} vars, ON={on_count}")

        num_vars = 0
        num_int_vars = 0
        num_constraints = 0

        for v in m.component_data_objects(
            pyo.Var,
            active=True,
            descend_into=True,
        ):
            if not v.is_fixed():
                num_vars += 1
                if v.is_binary() or v.is_integer():
                    num_int_vars += 1

        for c in m.component_data_objects(
            pyo.Constraint,
            active=True,
            descend_into=True,
        ):
            num_constraints += 1

        print(
            f"  [MODEL] total_vars={num_vars}, "
            f"integer_vars={num_int_vars}, "
            f"constraints={num_constraints}"
        )

        solver = create_solver(self.time_limit)

        if solver is None or not solver.available(exception_flag=False):
            raise RuntimeError("Gurobi Pyomo solver is unavailable.")

        solver.options["Threads"] = 0
        solver.options["OutputFlag"] = 1

        results = solve_model(solver, m, tee=True)

        tc = str(results.solver.termination_condition).lower()
        self.solve_time = time.time() - total_start

        if tc in ("optimal", "feasible", "locallyoptimal"):
            sol = model_obj.extract_results()
            print(f"[LLM] obj=${sol['objective']:,.0f}, time={self.solve_time:.1f}s")
            return sol
        else:
            print(f"[LLM] Infeasible: {tc}")
            return None

    def _call_llm(self, p, demand_total, reserve, wind_total, solar_total):
        system_prompt = self._build_system_prompt(p)
        user_prompt = self._build_user_prompt(demand_total, reserve, wind_total, solar_total)

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "timeout": 300.0,
        }

        kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.requests.create(**kwargs)
            print("  [LLM] Response received")
        except Exception as e:
            print(f"  [LLM] API error: {e}")
            return None

        self.token_usage = response.usage
        raw = response.choices[0].message.content

        G, T = p.num_generators, p.num_steps

        try:
            data = robust_json_parse(raw)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  [LLM] Parse failed: {e}")
            return None

        if "u" not in data:
            print("  [LLM] Missing 'u' key in response")
            return None

        u_arr = parse_u_strings(data["u"], G, T)
        return u_arr

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
4. Initial state compliance: generators with binding initial conditions must maintain their current state for the specified number of remaining steps.
5. Transmission capacity: power flow on each line is bounded by the thermal limit ({p.line_capacity[0]:.0f}MW). Buses with local demand exceeding local generation capacity depend on imports through adjacent lines.

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
