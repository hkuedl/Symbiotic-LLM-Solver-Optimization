"""Uncertainty-aware LLM sampling and solver-guided refinement."""

import numpy as np


class UncertaintyQuantifier:
    """UncertaintyQuantifier: uncertainty-aware llm sampling and solver-guided refinement."""

    def __init__(self, num_generators, num_steps):
        self.G = num_generators
        self.T = num_steps
        self.samples = []
        self.sample_probs = []
        self._cache = {}

    def _clear_cache(self):
        self._cache = {}

    def add_sample(self, u, logprob_probs=None):
        self.samples.append(u)
        self.sample_probs.append(logprob_probs)
        self._clear_cache()

    @property
    def N(self):
        return len(self.samples)

    @property
    def has_logprobs(self):
        return any(p is not None for p in self.sample_probs)

    def compute_empirical_probability(self):
        if "p_hat" in self._cache:
            return self._cache["p_hat"]
        N = self.N
        p = np.zeros((self.G, self.T))
        for u in self.samples:
            for g in range(self.G):
                for t in range(self.T):
                    p[g, t] += u[g][t]
        p /= N
        self._cache["p_hat"] = p
        return p

    def compute_predictive_entropy(self):
        if "H_pred" in self._cache:
            return self._cache["H_pred"]
        p = self.compute_empirical_probability()
        eps = 1e-10
        pc = np.clip(p, eps, 1 - eps)
        H = -pc * np.log(pc) - (1 - pc) * np.log(1 - pc)
        self._cache["H_pred"] = H
        return H

    def compute_aleatoric_entropy(self, window_size=3):
        """Use token probabilities when available; otherwise use temporal smoothing."""
        if "H_alea" in self._cache:
            return self._cache["H_alea"]
        eps = 1e-10
        N = self.N
        H_alea = np.zeros((self.G, self.T))
        for idx, u in enumerate(self.samples):
            u_arr = np.array(u, dtype=float)
            probs = self.sample_probs[idx]
            for g in range(self.G):
                for t in range(self.T):
                    if probs is not None and (g, t) in probs:
                        p_i = probs[(g, t)]
                    else:
                        t_lo = max(0, t - window_size)
                        t_hi = min(self.T, t + window_size + 1)
                        p_i = float(np.mean(u_arr[g, t_lo:t_hi]))
                    p_i = np.clip(p_i, eps, 1 - eps)
                    h = -p_i * np.log(p_i) - (1 - p_i) * np.log(1 - p_i)
                    H_alea[g, t] += h
        H_alea /= N
        self._cache["H_alea"] = H_alea
        return H_alea

    def compute_epistemic_uncertainty(self, window_size=3):
        """Return nonnegative predictive entropy minus aleatoric entropy."""
        if "I" in self._cache:
            return self._cache["I"]
        H_pred = self.compute_predictive_entropy()
        H_alea = self.compute_aleatoric_entropy(window_size)
        I = np.maximum(H_pred - H_alea, 0.0)
        self._cache["I"] = I
        return I

    def select_variables_to_fix(self, strategy="epistemic_threshold", **kwargs):
        """Select consensus assignments using a threshold, percentile, or budget."""
        p = self.compute_empirical_probability()
        I = self.compute_epistemic_uncertainty()
        fixes = {}

        if strategy == "epistemic_threshold":
            tau_I = kwargs.get("tau_I", 0.1)
            for g in range(self.G):
                for t in range(self.T):
                    if I[g, t] < tau_I:
                        fixes[(g, t)] = int(round(p[g, t]))

        elif strategy == "epistemic_percentile":
            alpha = kwargs.get("alpha", 0.5)
            threshold = np.percentile(I.flatten(), alpha * 100)
            for g in range(self.G):
                for t in range(self.T):
                    if I[g, t] <= threshold:
                        fixes[(g, t)] = int(round(p[g, t]))

        elif strategy == "epistemic_budget":
            beta = kwargs.get("beta", 0.3)
            all_vars = []
            for g in range(self.G):
                for t in range(self.T):
                    all_vars.append((I[g, t], g, t))
            all_vars.sort(key=lambda x: x[0])
            I_total = np.sum(I)
            budget = I_total * beta
            cumulative = 0.0
            for i_val, g, t in all_vars:
                if cumulative + i_val > budget:
                    break
                fixes[(g, t)] = int(round(p[g, t]))
                cumulative += i_val
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return fixes

    def summary(self):
        p = self.compute_empirical_probability()
        H = self.compute_predictive_entropy()
        H_alea = self.compute_aleatoric_entropy()
        I = self.compute_epistemic_uncertainty()
        total = self.G * self.T
        print(f"{'=' * 60}")
        print(f"  UQ SUMMARY  (N={self.N}, logprobs={self.has_logprobs})")
        print(f"{'=' * 60}")
        print(f"  Generators: {self.G}, Steps: {self.T}, Total vars: {total}")
        print(f"  Unanimous ON  (p=1):  {int(np.sum(p == 1)):5d} / {total}")
        print(f"  Unanimous OFF (p=0):  {int(np.sum(p == 0)):5d} / {total}")
        print(f"  Mixed (0<p<1):        {int(np.sum((p > 0) & (p < 1))):5d} / {total}")
        print(f"  H_pred:  mean={np.mean(H):.4f}, max={np.max(H):.4f}")
        print(f"  H_alea:  mean={np.mean(H_alea):.4f}, max={np.max(H_alea):.4f}")
        print(f"  I (epistemic): mean={np.mean(I):.4f}, max={np.max(I):.4f}, sum={np.sum(I):.2f}")
        print(
            f"  I=0: {int(np.sum(I < 1e-10)):5d} / {total} ({np.sum(I < 1e-10) / total * 100:.1f}%)"
        )
