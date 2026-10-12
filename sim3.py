"""
First-Price Sealed-Bid Auction -- Two-Round Qualification Mechanism
=======================================================================

Builds on sim2.py's asymmetric, random-unknown-N setting by adding a
two-round bidding mechanism with NO known theoretical equilibrium. This is
a "quick first pass" -- the mechanism as specified doesn't obviously admit
a clean equilibrium, and the docstring/comments below flag exactly where
that shows up in the learning dynamics. A later constraint may fix this.

Mechanism
---------
1. Round 1 (screening, NON-BINDING): every one of the N participating
   bidders submits a sealed bid. These bids are only used to rank
   entrants -- they are not paid and do not by themselves determine a
   winner.
2. Only the top `n_advance` bidders (by round-1 bid) advance to round 2.
   Everyone else is eliminated and earns 0.
3. Round 2 (binding): the advancing bidders submit a second sealed bid,
   with the constraint that it cannot be below their own round-1 bid.
   Highest round-2 bid among the qualifiers wins and pays that bid --
   otherwise exactly the first-price sealed-bid rules from sim1/sim2.

Why this likely has no clean equilibrium (yet)
-------------------------------------------------
Round-1 bids are payoff-relevant even though they are never paid: bidding
higher improves your odds of qualifying, but locks in a higher floor for
round 2 (where you're now effectively bidding against yourself as well as
the other qualifier(s)). There's nothing in the current mechanism pinning
down a "safe" floor to bid in round 1 -- bidding right at your value
guarantees qualification (if anyone can beat it) is meaningless since you
were going to win round 2 anyway at that price, but bidding *below* value
trades qualification risk for a cheaper floor. This general trade-off is
exactly why we're using simulation/self-play rather than solving for a
formula.

Learning algorithm
-------------------
Two separate contextual softmax policy-gradient tables per bidder type
(strong/weak), both indexed by a discretized value bin -- same style as
sim2.py:

- Round-2 (`TopUpBidder`) reuses the exact full-information trick from
  sim1/sim2: since round-2 bids are simultaneous among the qualifiers,
  the realized highest OTHER qualifier's bid is unaffected by a given
  qualifier's own choice, so the qualifier can evaluate the reward of
  every candidate "top-up fraction" s2 in [0, 1] -- where the actual bid
  is floor + s2 * (value - floor) -- in one shot each round.

- Round-1 (`QualificationBidder`) does NOT get this full-information
  shortcut: whether a given round-1 bid would have helped you qualify AND
  what would have happened for you in round 2 under a hypothetical
  round-1 bid both depend on a counterfactual replay we're not doing here
  (to keep this first pass simple). Instead it uses a plain single-action
  REINFORCE-with-baseline update (the "gradient bandit" from Sutton &
  Barto), fed the REALIZED downstream payoff of the whole two-round event
  (0 if eliminated or if qualified-but-lost, else the round-2 profit).
  This is much lower-signal/noisier than the round-2 update, so round-1
  learning is expected to be slower and possibly not fully converge --
  consistent with there being no known equilibrium to converge to.

Usage
-----
    py sim3.py
    py sim3.py --n-advance 2 --rounds 100000
    py sim3.py --strong-low 0.5 --weak-low 0.0 --p-strong 0.3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from sim1 import build_distribution, moving_average


# ---------------------------------------------------------------------------
# Distribution over the (unobserved) number of bidders
# ---------------------------------------------------------------------------

def build_n_distribution(name: str, **kwargs):
    """Frozen scipy.stats discrete distribution for the total number of bidders N."""
    name = name.lower()
    if name == "poisson":
        n_min = kwargs.get("n_min", 1)
        mean = kwargs.get("n_mean", 10.0)
        mu = max(mean - n_min, 1e-9)
        return stats.poisson(mu=mu, loc=n_min)
    if name == "discrete_uniform":
        n_min = kwargs.get("n_min", 5)
        n_max = kwargs.get("n_max", 15)
        return stats.randint(low=n_min, high=n_max + 1)
    raise ValueError(f"Unknown number-of-bidders distribution '{name}'")


# ---------------------------------------------------------------------------
# Round 2: full-information "top-up" bidder (binding, floor = round-1 bid)
# ---------------------------------------------------------------------------

class TopUpBidder:
    """
    Learns a bid TOP-UP function over the remaining surplus above a
    per-round floor: bid = floor + s(v) * (value - floor), s in [0, 1].

    Context (which preference row to use) is the bidder's discretized
    VALUE bin, same convention as sim2's ContextualPolicyGradientBidder.
    The action magnitude is scaled by the remaining surplus (value -
    floor) rather than by value itself, so reward(s) = (1 - s) if this
    bidder wins, else 0 -- exactly the same full-information update shape
    used everywhere else in sim1/sim2, just reinterpreted in terms of
    surplus instead of raw value.
    """

    def __init__(self, low: float, high: float, fraction_grid: np.ndarray,
                 num_value_bins: int = 21, alpha: float = 150.0):
        self.low = low
        self.high = high
        self.fraction_grid = fraction_grid
        self.alpha = alpha
        self.num_value_bins = num_value_bins
        self.bin_edges = np.linspace(low, high, num_value_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.preferences = np.zeros((num_value_bins, len(fraction_grid)))

    def bin_index_batch(self, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.bin_edges, values, side="right") - 1
        return np.clip(idx, 0, self.num_value_bins - 1)

    def bin_index(self, value: float) -> int:
        return int(self.bin_index_batch(np.array([value]))[0])

    @staticmethod
    def _softmax_rows(logits: np.ndarray) -> np.ndarray:
        h = logits - logits.max(axis=1, keepdims=True)
        exp_h = np.exp(h)
        return exp_h / exp_h.sum(axis=1, keepdims=True)

    def policy_row(self, bin_idx: int) -> np.ndarray:
        return self._softmax_rows(self.preferences[bin_idx][None, :])[0]

    def act_batch(self, values: np.ndarray, floors: np.ndarray, rng: np.random.Generator):
        """Returns (bids, bin_idxs, probs_matrix); floors is the round-1 bid per bidder."""
        bin_idxs = self.bin_index_batch(values)
        probs_matrix = self._softmax_rows(self.preferences[bin_idxs])

        cumprobs = np.cumsum(probs_matrix, axis=1)
        u = rng.random(len(values))
        frac_idxs = (cumprobs >= u[:, None]).argmax(axis=1)

        surplus = np.maximum(values - floors, 0.0)
        bids = floors + self.fraction_grid[frac_idxs] * surplus
        return bids, bin_idxs, probs_matrix

    def update_batch(self, bin_idxs: np.ndarray, values: np.ndarray, floors: np.ndarray,
                      highest_other: np.ndarray, probs_matrix: np.ndarray) -> None:
        surplus = np.maximum(values - floors, 1e-12)
        candidate_bids = floors[:, None] + self.fraction_grid[None, :] * surplus[:, None]
        rewards = np.where(candidate_bids > highest_other[:, None],
                            1.0 - self.fraction_grid[None, :], 0.0)
        baseline = np.sum(probs_matrix * rewards, axis=1, keepdims=True)
        delta = self.alpha * probs_matrix * (rewards - baseline)
        np.add.at(self.preferences, bin_idxs, delta)

    def expected_topup_fraction(self) -> np.ndarray:
        probs = np.array([self.policy_row(i) for i in range(self.num_value_bins)])
        return probs @ self.fraction_grid


# ---------------------------------------------------------------------------
# Round 1: non-binding, bandit-style qualification bidder
# ---------------------------------------------------------------------------

class QualificationBidder:
    """
    Learns a (non-binding) round-1 bid-shading function s(v) -> bid = s(v)*v,
    same table representation as sim2's ContextualPolicyGradientBidder, but
    updated via a single-action REINFORCE-with-baseline rule (Sutton &
    Barto's "gradient bandit") fed the REALIZED total downstream payoff of
    the whole two-round event -- see module docstring for why the
    full-information trick isn't used here.
    """

    def __init__(self, low: float, high: float, fraction_grid: np.ndarray,
                 num_value_bins: int = 21, alpha: float = 150.0, baseline_decay: float = 0.02):
        self.low = low
        self.high = high
        self.fraction_grid = fraction_grid
        self.alpha = alpha
        self.baseline_decay = baseline_decay
        self.num_value_bins = num_value_bins
        self.bin_edges = np.linspace(low, high, num_value_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.preferences = np.zeros((num_value_bins, len(fraction_grid)))
        self.baseline = np.zeros(num_value_bins)
        self._baseline_init = np.zeros(num_value_bins, dtype=bool)

    def bin_index_batch(self, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.bin_edges, values, side="right") - 1
        return np.clip(idx, 0, self.num_value_bins - 1)

    def bin_index(self, value: float) -> int:
        return int(self.bin_index_batch(np.array([value]))[0])

    @staticmethod
    def _softmax_rows(logits: np.ndarray) -> np.ndarray:
        h = logits - logits.max(axis=1, keepdims=True)
        exp_h = np.exp(h)
        return exp_h / exp_h.sum(axis=1, keepdims=True)

    def policy_row(self, bin_idx: int) -> np.ndarray:
        return self._softmax_rows(self.preferences[bin_idx][None, :])[0]

    def act_batch(self, values: np.ndarray, rng: np.random.Generator):
        """Returns (bids, bin_idxs, frac_idxs, probs_matrix)."""
        bin_idxs = self.bin_index_batch(values)
        probs_matrix = self._softmax_rows(self.preferences[bin_idxs])

        cumprobs = np.cumsum(probs_matrix, axis=1)
        u = rng.random(len(values))
        frac_idxs = (cumprobs >= u[:, None]).argmax(axis=1)

        bids = self.fraction_grid[frac_idxs] * values
        return bids, bin_idxs, frac_idxs, probs_matrix

    def update_batch(self, bin_idxs: np.ndarray, frac_idxs: np.ndarray,
                      rewards: np.ndarray, probs_matrix: np.ndarray) -> None:
        # Update each visited bin's baseline once, using the mean realized
        # reward of bidders who landed in that bin this round.
        for b in np.unique(bin_idxs):
            mask = bin_idxs == b
            mean_reward = rewards[mask].mean()
            if not self._baseline_init[b]:
                self.baseline[b] = mean_reward
                self._baseline_init[b] = True
            else:
                self.baseline[b] += self.baseline_decay * (mean_reward - self.baseline[b])

        advantage = rewards - self.baseline[bin_idxs]
        indicator = np.zeros_like(probs_matrix)
        indicator[np.arange(len(bin_idxs)), frac_idxs] = 1.0
        delta = self.alpha * advantage[:, None] * (indicator - probs_matrix)
        np.add.at(self.preferences, bin_idxs, delta)

    def expected_bid_function(self) -> np.ndarray:
        probs = np.array([self.policy_row(i) for i in range(self.num_value_bins)])
        expected_fraction = probs @ self.fraction_grid
        return expected_fraction * self.bin_centers


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class TypeConfig:
    low: float
    high: float
    dist_name: str = "uniform"
    dist_kwargs: dict = field(default_factory=dict)
    prob: float = 0.5  # probability a given bidder is drawn as this type


@dataclass
class SimulationConfig:
    strong: TypeConfig = field(default_factory=lambda: TypeConfig(low=0.5, high=1.0, prob=0.5))
    weak: TypeConfig = field(default_factory=lambda: TypeConfig(low=0.0, high=1.0, prob=0.5))
    n_dist_name: str = "poisson"
    n_dist_kwargs: dict = field(default_factory=lambda: {"n_mean": 10.0, "n_min": 1})
    n_advance: int = 2
    rounds: int = 20_000
    num_actions: int = 41
    num_value_bins: int = 21
    alpha: float = 150.0
    seed: int = 0


@dataclass
class SimulationResult:
    fraction_grid: np.ndarray
    strong_round1: QualificationBidder
    weak_round1: QualificationBidder
    strong_round2: TopUpBidder
    weak_round2: TopUpBidder
    strong_ref_bid1_history: np.ndarray
    weak_ref_bid1_history: np.ndarray
    qualify_rate_history: dict  # {"strong": array, "weak": array}
    win_rate_history: dict      # {"strong": array, "weak": array}
    mean_n_history: np.ndarray


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed)

    n_dist = build_n_distribution(cfg.n_dist_name, **cfg.n_dist_kwargs)
    strong_value_dist = build_distribution(cfg.strong.dist_name, cfg.strong.low, cfg.strong.high, **cfg.strong.dist_kwargs)
    weak_value_dist = build_distribution(cfg.weak.dist_name, cfg.weak.low, cfg.weak.high, **cfg.weak.dist_kwargs)

    fraction_grid = np.linspace(1.0 / cfg.num_actions, 1.0, cfg.num_actions)

    strong_round1 = QualificationBidder(cfg.strong.low, cfg.strong.high, fraction_grid, cfg.num_value_bins, cfg.alpha)
    weak_round1 = QualificationBidder(cfg.weak.low, cfg.weak.high, fraction_grid, cfg.num_value_bins, cfg.alpha)
    strong_round2 = TopUpBidder(cfg.strong.low, cfg.strong.high, fraction_grid, cfg.num_value_bins, cfg.alpha)
    weak_round2 = TopUpBidder(cfg.weak.low, cfg.weak.high, fraction_grid, cfg.num_value_bins, cfg.alpha)

    strong_ref_value = 0.5 * (cfg.strong.low + cfg.strong.high)
    weak_ref_value = 0.5 * (cfg.weak.low + cfg.weak.high)
    strong_ref_bin = strong_round1.bin_index(strong_ref_value)
    weak_ref_bin = weak_round1.bin_index(weak_ref_value)

    strong_ref_bid1_history = np.empty(cfg.rounds)
    weak_ref_bid1_history = np.empty(cfg.rounds)
    strong_qualify_history = np.empty(cfg.rounds)
    weak_qualify_history = np.empty(cfg.rounds)
    strong_win_history = np.empty(cfg.rounds)
    weak_win_history = np.empty(cfg.rounds)
    mean_n_history = np.empty(cfg.rounds)

    p_strong = cfg.strong.prob / (cfg.strong.prob + cfg.weak.prob)

    for t in range(cfg.rounds):
        strong_ref_bid1_history[t] = strong_round1.policy_row(strong_ref_bin) @ fraction_grid * strong_ref_value
        weak_ref_bid1_history[t] = weak_round1.policy_row(weak_ref_bin) @ fraction_grid * weak_ref_value

        n_total = max(int(n_dist.rvs(random_state=rng)), 1)
        mean_n_history[t] = n_total

        is_strong = rng.random(n_total) < p_strong
        n_strong = int(is_strong.sum())
        n_weak = n_total - n_strong

        values = np.empty(n_total)
        if n_strong:
            values[is_strong] = strong_value_dist.rvs(size=n_strong, random_state=rng)
        if n_weak:
            values[~is_strong] = weak_value_dist.rvs(size=n_weak, random_state=rng)

        # ---- Round 1: non-binding screening bid ----
        bids1 = np.empty(n_total)
        bin1_by_slot = np.empty(n_total, dtype=int)
        frac1_by_slot = np.empty(n_total, dtype=int)
        s_probs1 = w_probs1 = None

        if n_strong:
            s_bids1, s_bins1, s_fracs1, s_probs1 = strong_round1.act_batch(values[is_strong], rng)
            bids1[is_strong] = s_bids1
            bin1_by_slot[is_strong] = s_bins1
            frac1_by_slot[is_strong] = s_fracs1
        if n_weak:
            w_bids1, w_bins1, w_fracs1, w_probs1 = weak_round1.act_batch(values[~is_strong], rng)
            bids1[~is_strong] = w_bids1
            bin1_by_slot[~is_strong] = w_bins1
            frac1_by_slot[~is_strong] = w_fracs1

        # ---- Qualification: top n_advance by round-1 bid advance ----
        n_advance_actual = min(cfg.n_advance, n_total)
        qualifier_idx = np.argpartition(-bids1, n_advance_actual - 1)[:n_advance_actual]
        qualifies = np.zeros(n_total, dtype=bool)
        qualifies[qualifier_idx] = True

        # ---- Round 2: binding, bid >= own round-1 bid, among qualifiers only ----
        payoff = np.zeros(n_total)  # realized downstream utility, used as round-1 reward
        winner_among_qualifiers = None

        if n_advance_actual >= 1:
            q_values = values[qualifies]
            q_floors = bids1[qualifies]
            q_is_strong = is_strong[qualifies]

            q_bids2 = np.empty(n_advance_actual)
            q_bin2 = np.empty(n_advance_actual, dtype=int)
            q_probs2 = [None] * n_advance_actual

            q_strong_pos = np.where(q_is_strong)[0]
            q_weak_pos = np.where(~q_is_strong)[0]
            if len(q_strong_pos):
                b2, bin2, probs2 = strong_round2.act_batch(q_values[q_strong_pos], q_floors[q_strong_pos], rng)
                q_bids2[q_strong_pos] = b2
                q_bin2[q_strong_pos] = bin2
                for j, pos in enumerate(q_strong_pos):
                    q_probs2[pos] = probs2[j]
            if len(q_weak_pos):
                b2, bin2, probs2 = weak_round2.act_batch(q_values[q_weak_pos], q_floors[q_weak_pos], rng)
                q_bids2[q_weak_pos] = b2
                q_bin2[q_weak_pos] = bin2
                for j, pos in enumerate(q_weak_pos):
                    q_probs2[pos] = probs2[j]

            winner_local = int(np.argmax(q_bids2))
            if n_advance_actual > 1:
                sorted_bids2 = np.sort(q_bids2)[::-1]
                top1_2, top2_2 = sorted_bids2[0], sorted_bids2[1]
            else:
                top1_2, top2_2 = q_bids2[0], -np.inf
            highest_other2 = np.full(n_advance_actual, top1_2)
            highest_other2[winner_local] = top2_2

            q_reward = np.where(np.arange(n_advance_actual) == winner_local,
                                 q_values - q_bids2, 0.0)

            if len(q_strong_pos):
                probs2_strong = np.array([q_probs2[p] for p in q_strong_pos])
                strong_round2.update_batch(q_bin2[q_strong_pos], q_values[q_strong_pos], q_floors[q_strong_pos],
                                            highest_other2[q_strong_pos], probs2_strong)
            if len(q_weak_pos):
                probs2_weak = np.array([q_probs2[p] for p in q_weak_pos])
                weak_round2.update_batch(q_bin2[q_weak_pos], q_values[q_weak_pos], q_floors[q_weak_pos],
                                          highest_other2[q_weak_pos], probs2_weak)

            payoff[qualifier_idx] = q_reward
            winner_among_qualifiers = qualifier_idx[winner_local]

        # ---- Round-1 REINFORCE update, fed the realized total payoff ----
        reward1 = np.where(values > 0, payoff / np.maximum(values, 1e-12), 0.0)  # profit as fraction of value
        if n_strong:
            strong_round1.update_batch(bin1_by_slot[is_strong], frac1_by_slot[is_strong],
                                        reward1[is_strong], s_probs1)
        if n_weak:
            weak_round1.update_batch(bin1_by_slot[~is_strong], frac1_by_slot[~is_strong],
                                      reward1[~is_strong], w_probs1)

        strong_qualify_history[t] = qualifies[is_strong].mean() if n_strong else 0.0
        weak_qualify_history[t] = qualifies[~is_strong].mean() if n_weak else 0.0
        if winner_among_qualifiers is not None:
            strong_win_history[t] = 1.0 if is_strong[winner_among_qualifiers] else 0.0
            weak_win_history[t] = 1.0 if not is_strong[winner_among_qualifiers] else 0.0
        else:
            strong_win_history[t] = 0.0
            weak_win_history[t] = 0.0

    return SimulationResult(
        fraction_grid=fraction_grid,
        strong_round1=strong_round1,
        weak_round1=weak_round1,
        strong_round2=strong_round2,
        weak_round2=weak_round2,
        strong_ref_bid1_history=strong_ref_bid1_history,
        weak_ref_bid1_history=weak_ref_bid1_history,
        qualify_rate_history={"strong": strong_qualify_history, "weak": weak_qualify_history},
        win_rate_history={"strong": strong_win_history, "weak": weak_win_history},
        mean_n_history=mean_n_history,
    )


# ---------------------------------------------------------------------------
# Reporting / plotting
# ---------------------------------------------------------------------------

def summarize(cfg: SimulationConfig, res: SimulationResult) -> None:
    tail = max(1, cfg.rounds // 20)
    strong_ref_value = 0.5 * (cfg.strong.low + cfg.strong.high)
    weak_ref_value = 0.5 * (cfg.weak.low + cfg.weak.high)

    print("=" * 70)
    print("First-Price Sealed-Bid Auction | Two-Round Qualification Mechanism")
    print("=" * 70)
    print("No known equilibrium for this mechanism -- numbers below are a")
    print("SIMULATION-BASED ESTIMATE from self-play, and round-1 in particular")
    print("may not have converged (see module docstring).")
    print("-" * 70)
    print(f"Strong type: value ~ {cfg.strong.dist_name} on [{cfg.strong.low}, {cfg.strong.high}], "
          f"P(strong) = {cfg.strong.prob / (cfg.strong.prob + cfg.weak.prob):.2f}")
    print(f"Weak type:   value ~ {cfg.weak.dist_name} on [{cfg.weak.low}, {cfg.weak.high}], "
          f"P(weak)   = {cfg.weak.prob / (cfg.strong.prob + cfg.weak.prob):.2f}")
    print(f"Number-of-bidders distribution: {cfg.n_dist_name} {cfg.n_dist_kwargs}")
    print(f"n_advance (qualifiers to round 2): {cfg.n_advance}")
    print(f"Realized mean N over simulation: {res.mean_n_history.mean():.2f}")
    print(f"Rounds: {cfg.rounds}")
    print("-" * 70)
    print(f"Round-1 (screening) bid at strong ref. value v={strong_ref_value:.3f}: "
          f"{res.strong_ref_bid1_history[-1]:.4f}")
    print(f"Round-1 (screening) bid at weak ref. value   v={weak_ref_value:.3f}: "
          f"{res.weak_ref_bid1_history[-1]:.4f}")
    print(f"Strong qualify rate (last {tail} rounds): {res.qualify_rate_history['strong'][-tail:].mean():.4f}")
    print(f"Weak qualify rate (last {tail} rounds):   {res.qualify_rate_history['weak'][-tail:].mean():.4f}")
    print(f"Strong win rate (last {tail} rounds):     {res.win_rate_history['strong'][-tail:].mean():.4f}")
    print(f"Weak win rate (last {tail} rounds):       {res.win_rate_history['weak'][-tail:].mean():.4f}")
    print("=" * 70)


def plot_results(cfg: SimulationConfig, res: SimulationResult) -> None:
    window = max(1, cfg.rounds // 200)
    rounds_axis = np.arange(cfg.rounds)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Self-Play Estimate: Two-Round Qualification Mechanism")

    ax = axes[0, 0]
    ax.plot(rounds_axis, res.strong_ref_bid1_history, label="Strong: E[round-1 bid] at ref. value", lw=1)
    ax.plot(rounds_axis, res.weak_ref_bid1_history, label="Weak: E[round-1 bid] at ref. value", lw=1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Round-1 bid")
    ax.set_title("Round-1 (screening) bid convergence")
    ax.legend()

    ax = axes[0, 1]
    ma_sq = moving_average(res.qualify_rate_history["strong"], window)
    ma_wq = moving_average(res.qualify_rate_history["weak"], window)
    ax.plot(rounds_axis[: len(ma_sq)], ma_sq, label="Strong qualify rate", lw=1)
    ax.plot(rounds_axis[: len(ma_wq)], ma_wq, label="Weak qualify rate", lw=1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Qualify rate")
    ax.set_title("Qualification rate over time")
    ax.legend()

    ax = axes[1, 0]
    strong_b1 = res.strong_round1.expected_bid_function()
    weak_b1 = res.weak_round1.expected_bid_function()
    ax.plot(res.strong_round1.bin_centers, strong_b1, marker="o", ms=3, label="Strong round-1 bid b1(v)")
    ax.plot(res.weak_round1.bin_centers, weak_b1, marker="o", ms=3, label="Weak round-1 bid b1(v)")
    ax.plot([0, 1], [0, 1], ls=":", color="gray", label="Bid = value")
    ax.set_xlabel("Value v")
    ax.set_ylabel("Round-1 bid")
    ax.set_title("Learned round-1 (screening) bid function")
    ax.legend()

    ax = axes[1, 1]
    strong_s2 = res.strong_round2.expected_topup_fraction()
    weak_s2 = res.weak_round2.expected_topup_fraction()
    # Illustrative round-2 bid using this type's OWN round-1 bid function as the floor.
    strong_b2 = strong_b1 + strong_s2 * (res.strong_round2.bin_centers - strong_b1)
    weak_b2 = weak_b1 + weak_s2 * (res.weak_round2.bin_centers - weak_b1)
    ax.plot(res.strong_round2.bin_centers, strong_b2, marker="o", ms=3, label="Strong round-2 bid b2(v)")
    ax.plot(res.weak_round2.bin_centers, weak_b2, marker="o", ms=3, label="Weak round-2 bid b2(v)")
    ax.plot([0, 1], [0, 1], ls=":", color="gray", label="Bid = value")
    ax.set_xlabel("Value v")
    ax.set_ylabel("Round-2 bid")
    ax.set_title("Learned round-2 (final) bid function\n(using own round-1 bid as floor)")
    ax.legend()

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--high", type=float, default=1.0, help="Common upper bound of both value distributions")
    parser.add_argument("--strong-low", type=float, default=0.5, help="Lower bound of the strong type's value distribution")
    parser.add_argument("--weak-low", type=float, default=0.0, help="Lower bound of the weak type's value distribution")
    parser.add_argument("--strong-dist", dest="strong_dist_name", default="uniform", choices=["uniform", "beta", "truncnorm"])
    parser.add_argument("--weak-dist", dest="weak_dist_name", default="uniform", choices=["uniform", "beta", "truncnorm"])
    parser.add_argument("--strong-beta-a", type=float, default=2.0)
    parser.add_argument("--strong-beta-b", type=float, default=2.0)
    parser.add_argument("--weak-beta-a", type=float, default=2.0)
    parser.add_argument("--weak-beta-b", type=float, default=2.0)
    parser.add_argument("--p-strong", type=float, default=0.5, help="Probability a given bidder is the strong type")

    parser.add_argument("--n-dist", dest="n_dist_name", default="poisson", choices=["poisson", "discrete_uniform"])
    parser.add_argument("--n-mean", type=float, default=10.0)
    parser.add_argument("--n-min", type=int, default=None)
    parser.add_argument("--n-max", type=int, default=15)
    parser.add_argument("--n-advance", type=int, default=2, help="Number of bidders who advance to round 2")

    parser.add_argument("--rounds", type=int, default=20_000)
    parser.add_argument("--num-actions", type=int, default=41, help="Resolution of the bid-shading fraction grid")
    parser.add_argument("--num-value-bins", type=int, default=21, help="Resolution of the value discretization per type")
    parser.add_argument("--alpha", type=float, default=150.0, help="Policy-gradient learning rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    strong_kwargs = {"beta_a": args.strong_beta_a, "beta_b": args.strong_beta_b} if args.strong_dist_name == "beta" else {}
    weak_kwargs = {"beta_a": args.weak_beta_a, "beta_b": args.weak_beta_b} if args.weak_dist_name == "beta" else {}

    n_dist_kwargs = {}
    if args.n_dist_name == "poisson":
        n_dist_kwargs["n_mean"] = args.n_mean
        n_dist_kwargs["n_min"] = args.n_min if args.n_min is not None else 1
    else:
        n_dist_kwargs["n_min"] = args.n_min if args.n_min is not None else 5
        n_dist_kwargs["n_max"] = args.n_max

    cfg = SimulationConfig(
        strong=TypeConfig(low=args.strong_low, high=args.high, dist_name=args.strong_dist_name,
                           dist_kwargs=strong_kwargs, prob=args.p_strong),
        weak=TypeConfig(low=args.weak_low, high=args.high, dist_name=args.weak_dist_name,
                         dist_kwargs=weak_kwargs, prob=1.0 - args.p_strong),
        n_dist_name=args.n_dist_name,
        n_dist_kwargs=n_dist_kwargs,
        n_advance=args.n_advance,
        rounds=args.rounds,
        num_actions=args.num_actions,
        num_value_bins=args.num_value_bins,
        alpha=args.alpha,
        seed=args.seed,
    )
    cfg._no_plot = args.no_plot  # type: ignore[attr-defined]
    return cfg


def main() -> None:
    cfg = parse_args()
    res = run_simulation(cfg)
    summarize(cfg, res)
    if not getattr(cfg, "_no_plot", False):
        plot_results(cfg, res)


if __name__ == "__main__":
    main()
