"""
First-Price Sealed-Bid Auction -- Asymmetric Bidders, Random Unknown N
=========================================================================

This drops the two assumptions of sim1.py that made a closed-form
equilibrium available (symmetry and a known/fixed n), and instead uses
self-play learning to *estimate* where an equilibrium (if one exists)
sits.

Setup
-----
- The total number of bidders N in a round is a random variable, drawn
  fresh each round from a commonly-known distribution (Poisson or
  discrete-uniform, reused from sim2's original random-N model). Nobody
  observes the realized N -- only its distribution is common knowledge.
- Each of the N bidders is independently assigned a TYPE:
    * "strong" bidders draw values from F_strong on [low_strong, high]
    * "weak"   bidders draw values from F_weak   on [low_weak,   high]
  with low_strong > low_weak and a common upper bound `high`. This is
  the classic asymmetric IPV first-price setting (Maskin & Riley 2000;
  Lebrun 1999) -- strong bidders are stochastically favored (their value
  distribution has no mass below low_strong), and in general there is no
  closed-form bidding equilibrium once F_strong != F_weak.

Why self-play instead of solving for an equilibrium
------------------------------------------------------
With asymmetric bidders there is generally no tractable formula for the
equilibrium bid functions b_strong(v), b_weak(v). So instead of handing
rivals a known equilibrium strategy (as in sim1/sim2), BOTH types learn
their own bid function simultaneously, purely from repeated play against
each other. If the learning process settles down, the resulting pair of
learned bid functions is our simulation-based estimate of the (Bayesian
Nash) equilibrium.

Learning algorithm
-------------------
Each type maintains a table of softmax action-preferences indexed by a
discretized VALUE BIN. The action space is a "bid-shading fraction"
s in (0, 1], so a bidder with value v bids b = s * v (this automatically
rules out ever bidding above one's own value, and lets one fraction grid
be shared by both types regardless of their support).

Sealed bids are simultaneous and independent, so -- exactly as in sim1 --
the highest bid among a bidder's OPPONENTS in a given round would be the
same no matter what that bidder bid. Given that "highest other bid" M_i,
a bidder can evaluate the profit (as a fraction of its own value) it
*would have* earned at every shading fraction on the grid:

    reward(s) = (1 - s) if s * v > M_i else 0

and take an exact single-sample softmax policy-gradient step on its
type's preference row for its value bin. Because several bidders of the
same type can appear in the same round (N is random), each one
contributes an update to the shared per-type table that round.

This is the same core learning rule as sim1.py/sim2.py, just applied
per-(type, value-bin) context instead of to a single fixed value.

Usage
-----
    py sim2.py
    py sim2.py --strong-low 0.5 --weak-low 0.0 --p-strong 0.3
    py sim2.py --strong-dist beta --strong-beta-a 3 --strong-beta-b 1
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
        mean = kwargs.get("n_mean", 50.0)
        mu = max(mean - n_min, 1e-9)
        return stats.poisson(mu=mu, loc=n_min)
    if name == "discrete_uniform":
        n_min = kwargs.get("n_min", 30)
        n_max = kwargs.get("n_max", 70)
        return stats.randint(low=n_min, high=n_max + 1)
    raise ValueError(f"Unknown number-of-bidders distribution '{name}'")


# ---------------------------------------------------------------------------
# Contextual (value-binned) policy-gradient bidder
# ---------------------------------------------------------------------------

class ContextualPolicyGradientBidder:
    """
    Learns a bid-shading function s(v) -> bid = s(v) * v over a value
    range [low, high], represented as a table of softmax preferences with
    one row per discretized value bin and one column per candidate
    shading fraction (shared fraction grid across all bins/types).

    See module docstring for the update rule; it is the same
    full-information softmax policy-gradient step used in sim1.py,
    applied independently to whichever value-bin row is relevant.
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

    def bin_index(self, value: float) -> int:
        idx = np.searchsorted(self.bin_edges, value, side="right") - 1
        return int(np.clip(idx, 0, self.num_value_bins - 1))

    def policy_row(self, bin_idx: int) -> np.ndarray:
        h = self.preferences[bin_idx]
        h = h - h.max()
        exp_h = np.exp(h)
        return exp_h / exp_h.sum()

    def act(self, value: float, rng: np.random.Generator) -> tuple[float, int, int]:
        bin_idx = self.bin_index(value)
        probs = self.policy_row(bin_idx)
        frac_idx = int(rng.choice(len(self.fraction_grid), p=probs))
        bid = self.fraction_grid[frac_idx] * value
        return bid, bin_idx, frac_idx

    def update(self, bin_idx: int, value: float, highest_other_bid: float) -> None:
        candidate_bids = self.fraction_grid * value
        rewards = np.where(candidate_bids > highest_other_bid, 1.0 - self.fraction_grid, 0.0)
        probs = self.policy_row(bin_idx)
        baseline = np.dot(probs, rewards)
        self.preferences[bin_idx] += self.alpha * probs * (rewards - baseline)

    # -- Vectorized batch versions, used by the simulation loop for speed --

    def bin_index_batch(self, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.bin_edges, values, side="right") - 1
        return np.clip(idx, 0, self.num_value_bins - 1)

    @staticmethod
    def _softmax_rows(logits: np.ndarray) -> np.ndarray:
        h = logits - logits.max(axis=1, keepdims=True)
        exp_h = np.exp(h)
        return exp_h / exp_h.sum(axis=1, keepdims=True)

    def act_batch(self, values: np.ndarray, rng: np.random.Generator):
        """
        Vectorized version of `act` for many bidders of this type at once
        (all acting on the same, pre-update preference table -- consistent
        with simultaneous sealed bidding).

        Returns (bids, bin_idxs, probs_matrix); probs_matrix is reused by
        `update_batch` so the update baseline is computed against the same
        pre-round policy that was actually used to bid.
        """
        bin_idxs = self.bin_index_batch(values)
        probs_matrix = self._softmax_rows(self.preferences[bin_idxs])  # (k, num_actions)

        cumprobs = np.cumsum(probs_matrix, axis=1)
        u = rng.random(len(values))
        frac_idxs = (cumprobs >= u[:, None]).argmax(axis=1)

        bids = self.fraction_grid[frac_idxs] * values
        return bids, bin_idxs, probs_matrix

    def update_batch(self, bin_idxs: np.ndarray, values: np.ndarray,
                      highest_other: np.ndarray, probs_matrix: np.ndarray) -> None:
        """Batch policy-gradient update for many bidders of this type at once."""
        candidate_bids = self.fraction_grid[None, :] * values[:, None]          # (k, num_actions)
        rewards = np.where(candidate_bids > highest_other[:, None],
                            1.0 - self.fraction_grid[None, :], 0.0)             # (k, num_actions)
        baseline = np.sum(probs_matrix * rewards, axis=1, keepdims=True)        # (k, 1)
        delta = self.alpha * probs_matrix * (rewards - baseline)                # (k, num_actions)
        np.add.at(self.preferences, bin_idxs, delta)

    def expected_bid_function(self) -> np.ndarray:
        """Expected bid E[s | bin] * bin_center, for every value bin."""
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
    rounds: int = 20_000
    num_actions: int = 41
    num_value_bins: int = 21
    alpha: float = 150.0
    seed: int = 0


@dataclass
class SimulationResult:
    fraction_grid: np.ndarray
    strong_learner: ContextualPolicyGradientBidder
    weak_learner: ContextualPolicyGradientBidder
    strong_ref_bid_history: np.ndarray
    weak_ref_bid_history: np.ndarray
    strong_win_history: np.ndarray
    weak_win_history: np.ndarray
    mean_n_history: np.ndarray


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed)

    n_dist = build_n_distribution(cfg.n_dist_name, **cfg.n_dist_kwargs)
    strong_value_dist = build_distribution(cfg.strong.dist_name, cfg.strong.low, cfg.strong.high, **cfg.strong.dist_kwargs)
    weak_value_dist = build_distribution(cfg.weak.dist_name, cfg.weak.low, cfg.weak.high, **cfg.weak.dist_kwargs)

    fraction_grid = np.linspace(1.0 / cfg.num_actions, 1.0, cfg.num_actions)

    strong_learner = ContextualPolicyGradientBidder(
        cfg.strong.low, cfg.strong.high, fraction_grid, cfg.num_value_bins, cfg.alpha
    )
    weak_learner = ContextualPolicyGradientBidder(
        cfg.weak.low, cfg.weak.high, fraction_grid, cfg.num_value_bins, cfg.alpha
    )

    strong_ref_value = 0.5 * (cfg.strong.low + cfg.strong.high)
    weak_ref_value = 0.5 * (cfg.weak.low + cfg.weak.high)
    strong_ref_bin = strong_learner.bin_index(strong_ref_value)
    weak_ref_bin = weak_learner.bin_index(weak_ref_value)

    strong_ref_bid_history = np.empty(cfg.rounds)
    weak_ref_bid_history = np.empty(cfg.rounds)
    strong_win_history = np.empty(cfg.rounds)
    weak_win_history = np.empty(cfg.rounds)
    mean_n_history = np.empty(cfg.rounds)

    p_strong = cfg.strong.prob / (cfg.strong.prob + cfg.weak.prob)

    for t in range(cfg.rounds):
        strong_ref_bid_history[t] = strong_learner.policy_row(strong_ref_bin) @ fraction_grid * strong_ref_value
        weak_ref_bid_history[t] = weak_learner.policy_row(weak_ref_bin) @ fraction_grid * weak_ref_value

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

        bids = np.empty(n_total)
        bin_idxs_by_slot = np.empty(n_total, dtype=int)  # bin idx within each bidder's OWN type table

        if n_strong:
            s_bids, s_bins, s_probs = strong_learner.act_batch(values[is_strong], rng)
            bids[is_strong] = s_bids
            bin_idxs_by_slot[is_strong] = s_bins
        if n_weak:
            w_bids, w_bins, w_probs = weak_learner.act_batch(values[~is_strong], rng)
            bids[~is_strong] = w_bids
            bin_idxs_by_slot[~is_strong] = w_bins

        winner_idx = int(np.argmax(bids))
        if n_total > 1:
            sorted_bids = np.sort(bids)[::-1]
            top1, top2 = sorted_bids[0], sorted_bids[1]
        else:
            top1, top2 = bids[0], -np.inf
        highest_other = np.full(n_total, top1)
        highest_other[winner_idx] = top2

        if n_strong:
            strong_learner.update_batch(bin_idxs_by_slot[is_strong], values[is_strong],
                                         highest_other[is_strong], s_probs)
        if n_weak:
            weak_learner.update_batch(bin_idxs_by_slot[~is_strong], values[~is_strong],
                                       highest_other[~is_strong], w_probs)

        strong_win_history[t] = 1.0 if is_strong[winner_idx] else 0.0
        weak_win_history[t] = 1.0 if not is_strong[winner_idx] else 0.0

    return SimulationResult(
        fraction_grid=fraction_grid,
        strong_learner=strong_learner,
        weak_learner=weak_learner,
        strong_ref_bid_history=strong_ref_bid_history,
        weak_ref_bid_history=weak_ref_bid_history,
        strong_win_history=strong_win_history,
        weak_win_history=weak_win_history,
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
    print("First-Price Sealed-Bid Auction | Asymmetric Bidders, Random N")
    print("=" * 70)
    print("No closed-form equilibrium generally exists for this setting --")
    print("the numbers below are a SIMULATION-BASED ESTIMATE from self-play.")
    print("-" * 70)
    print(f"Strong type: value ~ {cfg.strong.dist_name} on [{cfg.strong.low}, {cfg.strong.high}], "
          f"P(strong) = {cfg.strong.prob / (cfg.strong.prob + cfg.weak.prob):.2f}")
    print(f"Weak type:   value ~ {cfg.weak.dist_name} on [{cfg.weak.low}, {cfg.weak.high}], "
          f"P(weak)   = {cfg.weak.prob / (cfg.strong.prob + cfg.weak.prob):.2f}")
    print(f"Number-of-bidders distribution: {cfg.n_dist_name} {cfg.n_dist_kwargs}")
    print(f"Realized mean N over simulation: {res.mean_n_history.mean():.2f}")
    print(f"Rounds: {cfg.rounds}")
    print("-" * 70)
    print(f"Learned bid at strong reference value v={strong_ref_value:.3f}: "
          f"{res.strong_ref_bid_history[-1]:.4f} (shade = {res.strong_ref_bid_history[-1] / strong_ref_value:.4f})")
    print(f"Learned bid at weak reference value   v={weak_ref_value:.3f}: "
          f"{res.weak_ref_bid_history[-1]:.4f} (shade = {res.weak_ref_bid_history[-1] / weak_ref_value:.4f})")
    print(f"Strong win rate (last {tail} rounds): {res.strong_win_history[-tail:].mean():.4f}")
    print(f"Weak win rate (last {tail} rounds):   {res.weak_win_history[-tail:].mean():.4f}")
    print("=" * 70)


def plot_results(cfg: SimulationConfig, res: SimulationResult) -> None:
    window = max(1, cfg.rounds // 200)
    rounds_axis = np.arange(cfg.rounds)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Self-Play Estimate of Asymmetric First-Price Auction Equilibrium")

    ax = axes[0, 0]
    ax.plot(rounds_axis, res.strong_ref_bid_history, label="Strong: E[bid] at ref. value", lw=1)
    ax.plot(rounds_axis, res.weak_ref_bid_history, label="Weak: E[bid] at ref. value", lw=1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Bid")
    ax.set_title("Convergence at reference values")
    ax.legend()

    ax = axes[0, 1]
    ma_strong_win = moving_average(res.strong_win_history, window)
    ma_weak_win = moving_average(res.weak_win_history, window)
    ax.plot(rounds_axis[: len(ma_strong_win)], ma_strong_win, label="Strong win rate", lw=1)
    ax.plot(rounds_axis[: len(ma_weak_win)], ma_weak_win, label="Weak win rate", lw=1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Win rate")
    ax.set_title("Win rate over time (per type)")
    ax.legend()

    ax = axes[1, 0]
    strong_bid_fn = res.strong_learner.expected_bid_function()
    weak_bid_fn = res.weak_learner.expected_bid_function()
    ax.plot(res.strong_learner.bin_centers, strong_bid_fn, marker="o", ms=3, label="Learned b_strong(v)")
    ax.plot(res.weak_learner.bin_centers, weak_bid_fn, marker="o", ms=3, label="Learned b_weak(v)")
    ax.plot([0, 1], [0, 1], ls=":", color="gray", label="Bid = value (no shading)")
    ax.set_xlabel("Value v")
    ax.set_ylabel("Bid")
    ax.set_title("Estimated equilibrium bid functions")
    ax.legend()

    ax = axes[1, 1]
    strong_shade = strong_bid_fn / np.maximum(res.strong_learner.bin_centers, 1e-9)
    weak_shade = weak_bid_fn / np.maximum(res.weak_learner.bin_centers, 1e-9)
    ax.plot(res.strong_learner.bin_centers, strong_shade, marker="o", ms=3, label="Strong shading s(v)=bid/v")
    ax.plot(res.weak_learner.bin_centers, weak_shade, marker="o", ms=3, label="Weak shading s(v)=bid/v")
    ax.set_xlabel("Value v")
    ax.set_ylabel("Shading fraction")
    ax.set_title("Learned bid-shading fraction")
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
