"""Shared parsing and schedule postprocessing utilities."""

import json
import re

import numpy as np


def robust_json_parse(raw):
    """Recover a JSON object from complete or truncated LLM output."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```\w*\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    if ".repeat(" in cleaned or "repeat(" in cleaned:
        cleaned = re.sub(r'"[01]"\.repeat\(\d+\)', lambda m: None, cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    if start != -1:
        partial = cleaned[start:]
        partial = re.sub(r"[,\s]+$", "", partial)
        partial = re.sub(r",\s*$", "", partial)
        open_sq = partial.count("[") - partial.count("]")
        open_cr = partial.count("{") - partial.count("}")
        if open_sq > 0 or open_cr > 0:
            try:
                return json.loads(partial + "]" * max(0, open_sq) + "}" * max(0, open_cr))
            except json.JSONDecodeError:
                pass

    cleaned_tail = re.sub(r"[,\s]+$", "", cleaned)
    open_sq = cleaned_tail.count("[") - cleaned_tail.count("]")
    open_cr = cleaned_tail.count("{") - cleaned_tail.count("}")
    suffix = "]" * max(0, open_sq) + "}" * max(0, open_cr)
    try:
        return json.loads(cleaned_tail + suffix)
    except json.JSONDecodeError:
        pass

    u_match = re.search(r'"u"\s*:\s*(\[[\s\S]*)', cleaned)
    if u_match:
        u_str = u_match.group(1).strip()
        u_str = re.sub(r"[,\s]+$", "", u_str)
        open_sq = u_str.count("[") - u_str.count("]")
        u_str += "]" * max(0, open_sq)
        try:
            return {"u": json.loads(u_str)}
        except json.JSONDecodeError:
            for cut in range(len(u_str) - 1, 0, -1):
                if u_str[cut] in "],":
                    candidate = u_str[: cut + 1]
                    candidate = re.sub(r",\s*$", "", candidate)
                    o = candidate.count("[") - candidate.count("]")
                    candidate += "]" * max(0, o)
                    try:
                        return {"u": json.loads(candidate)}
                    except json.JSONDecodeError:
                        continue

    raise ValueError(f"Cannot parse. First 300 chars: {raw[:300]}")


def _ragged_to_2d(raw_list, target, fill):
    G, T = target
    result = np.full(target, fill, dtype=float)
    if not isinstance(raw_list, list):
        return result
    for g in range(min(len(raw_list), G)):
        if not isinstance(raw_list[g], list):
            if isinstance(raw_list[g], (int, float)):
                result[g, :] = int(round(raw_list[g]))
            continue
        for t in range(min(len(raw_list[g]), T)):
            val = raw_list[g][t]
            if isinstance(val, (int, float)):
                result[g, t] = val
    return result


def to_2d(arr_or_list, target, fill=0):
    G, T = target
    if isinstance(arr_or_list, np.ndarray):
        arr = arr_or_list
    else:
        return _ragged_to_2d(arr_or_list, target, fill)

    while arr.ndim > 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.flatten()[: G * T].reshape(G, T)
            break
    while arr.ndim < 2:
        arr = np.expand_dims(arr, 0)

    if arr.shape != target:
        if arr.shape == (T, G):
            arr = arr.T

    if arr.shape != target:
        result = np.full(target, fill, dtype=float)
        sg, st = min(arr.shape[0], G), min(arr.shape[1], T)
        result[:sg, :st] = arr[:sg, :st]
        arr = result

    return arr


def parse_u_strings(u_raw, G, T):
    if not isinstance(u_raw, list) or len(u_raw) == 0:
        return np.zeros((G, T), dtype=int)

    if isinstance(u_raw[0], str):
        result = np.zeros((G, T), dtype=int)
        for g in range(min(len(u_raw), G)):
            s = u_raw[g]
            for t in range(min(len(s), T)):
                if s[t] == "1":
                    result[g, t] = 1
        return result

    return to_2d(u_raw, (G, T), fill=0)


def postprocess(fixes, p):
    for g in range(p.num_generators):
        t = 0
        while t < p.num_steps:
            if fixes.get((g, t)) == 1:
                t_start = t
                while t < p.num_steps and fixes.get((g, t)) == 1:
                    t += 1
                if t - t_start < p.min_up_time[g]:
                    for tt in range(t_start, t):
                        fixes.pop((g, tt), None)
            else:
                t += 1

    for g in range(p.num_generators):
        t = 0
        while t < p.num_steps:
            if fixes.get((g, t)) == 0:
                t_start = t
                while t < p.num_steps and fixes.get((g, t)) == 0:
                    t += 1
                if t - t_start < p.min_down_time[g]:
                    for tt in range(t_start, t):
                        fixes.pop((g, tt), None)
            else:
                t += 1

    return fixes


def fix_variables(m, fixes, p):
    for (g, t), val in fixes.items():
        m.u[g, t].fix(val)
        prev = p.u0[g] if t == 0 else fixes.get((g, t - 1), None)
        if prev is not None:
            m.v[g, t].fix(max(0, val - prev))
            m.w[g, t].fix(max(0, prev - val))


def print_llm_usage(token_usage, llm_time):
    print(f"  [LLM] Wall time     = {llm_time:.1f}s")
    if token_usage is None:
        return
