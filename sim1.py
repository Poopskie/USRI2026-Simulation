"""
First-Price Sealed-Bid Auction -- Independent Private Values (IPV)
====================================================================

Setup
-----
- n = 50 risk-neutral bidders participate in a first-price sealed-bid
  auction.
- n - 1 "market" bidders draw a fresh private value each round from a
  common, known distribution F, and bid according to the symmetric
  risk-neutral Bayesian-Nash equilibrium (BNE) strategy for F and n.
- One "focal" bidder has a FIXED, known private value v0 (it does not
  redraw each round). It does NOT know the equilibrium bidding formula.
  Instead, it learns how to bid purely from repeated play.

Learning algorithm
-------------------
The focal bidder uses softmax policy-gradient ascent over a discretized
grid of candidate bids in [0, v0] (bidding above one's own value is
always weakly dominated for a risk-neutral bidder, so the grid is capped
there).

Sealed bids are submitted simultaneously and independently, so the
realized highest rival bid M_t in a given round would have been exactly
the same regardless of which bid the focal bidder chose. Assuming this
highest competing bid becomes observable after the round closes (as with
a posted clearing price, common in ad-exchange and public-auction
records), the bidder can, each round, evaluate the reward it *would have*
earned at every candidate bid on its grid: R(b) = (v0 - b) * 1{b > M_t}.
This gives an exact single-sample estimate of the softmax policy
gradient of expected utility each round, so the policy's probability
mass gradually concentrates on the utility-maximizing bid. (With n=50
rivals, the win probability at any bid is small, so plain single-arm
bandit feedback -- observing only the outcome of the bid actually
played -- would need far more rounds to learn anything.)

Because all n - 1 rivals play the true BNE strategy, the best response
for the focal bidder is exactly the same equilibrium bid function
evaluated at v0 -- i.e. b*(v0) = argmax_b (v0 - b) * P(win | b). This
theoretical benchmark is computed numerically and used to check whether
the learned strategy converges to it.

Usage
-----
    py sim1.py
    py sim1.py --n 50 --v0 0.9 --rounds 20000 --dist uniform
    py sim1.py --dist beta --beta-a 2 --beta-b 5 --v0 0.85

Note: with n=50 rivals, the maximum of 49 i.i.d. draws concentrates near
the top of the support, so a focal value well below the upper end of the
support will rarely be able to win at all. Pick v0 reasonably high (or
reduce n) if you want a non-negligible, quickly-learnable win rate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.integrate import cumulative_trapezoid

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def build_distribution(name: str, low: float, high: float, **kwargs):
    """Return a frozen scipy.stats distribution supported on [low, high]."""
    name = name.lower()
    if name == "uniform":
        return stats.uniform(loc=low, scale=high - low)
    if name == "beta":
        a = kwargs.get("beta_a", 2.0)
        b = kwargs.get("beta_b", 2.0)
        return stats.beta(a, b, loc=low, scale=high - low)
    if name == "truncnorm":
        mu = kwargs.get("norm_mu", (low + high) / 2)
        sigma = kwargs.get("norm_sigma", (high - low) / 6)
        a = (low - mu) / sigma
        b = (high - mu) / sigma
        return stats.truncnorm(a, b, loc=mu, scale=sigma)
    raise ValueError(f"Unknown distribution '{name}'")


class EquilibriumBidder:
    """
    Symmetric risk-neutral BNE bid function for a first-price sealed-bid
    auction with `n` bidders whose values are i.i.d. draws from `dist` on
    [low, high]:

        b(v) = v - (1 / F(v)^(n-1)) * integral_{low}^{v} F(x)^(n-1) dx

    The integral is evaluated once on a fine grid and the bid function is
    then obtained anywhere via linear interpolation, which is fast enough
    to call thousands of times per round.
    """

    def __init__(self, dist, n: int, low: float, high: float, grid_size: int = 4000):
        self.dist = dist
        self.n = n
        self.low = low
        self.high = high

        self.grid = np.linspace(low, high, grid_size)
        F = dist.cdf(self.grid)
        F = np.clip(F, 1e-300, None)  # avoid 0^(n-1) issues at the left edge
        F_pow = F ** (n - 1)

        integral = cumulative_trapezoid(F_pow, self.grid, initial=0.0)
        # F_pow can underflow to exactly 0.0 for a range near the left edge
        # of skewed distributions (e.g. beta(3, 1) with large n), which would
        # otherwise produce 0/0 = nan there. In that regime both v and the
        # true equilibrium bid are close to `low` anyway, so falling back to
        # `low` is a harmless approximation -- and, crucially, avoids poisoning
        # the monotonic np.maximum.accumulate pass below (NaN propagates
        # forward through accumulate once introduced).
        safe = F_pow > 0
        bid_grid = np.full_like(self.grid, low)
        bid_grid[safe] = self.grid[safe] - integral[safe] / F_pow[safe]
        # At v = low, F_pow -> 0 and the ratio -> low (bidder's own value);
        # fix any numerical artifact at the boundary.
        bid_grid[0] = low
        bid_grid = np.maximum.accumulate(bid_grid)  # bid function is monotonic
        self.bid_grid = bid_grid

    def bid(self, values: np.ndarray) -> np.ndarray:
        """Vectorized equilibrium bid for an array of private values."""
        return np.interp(values, self.grid, self.bid_grid)

    def win_prob(self, bid: float) -> float:
        """P(this bid beats n-1 rivals bidding the equilibrium strategy)."""
        v_equiv = np.interp(bid, self.bid_grid, self.grid)
        return float(self.dist.cdf(v_equiv) ** (self.n - 1))

    def best_response(self, v0: float, bid_candidates: np.ndarray) -> tuple[np.ndarray, int, float]:
        """Expected utility of value-v0 bidder at each candidate bid."""
        win_probs = np.array([self.win_prob(b) for b in bid_candidates])
        expected_utility = (v0 - bid_candidates) * win_probs
        best_idx = int(np.argmax(expected_utility))
        return expected_utility, best_idx, win_probs[best_idx]


# ---------------------------------------------------------------------------
# Gradient-bandit learner
# ---------------------------------------------------------------------------

class PolicyGradientBidder:
    """
    Learns a bidding strategy over a discretized bid grid via softmax
    policy-gradient ascent on expected utility.

    Feedback model
    ---------------
    Bids in a sealed-bid auction are submitted simultaneously and
    independently, so the realized highest rival bid M_t in a given round
    would have been exactly the same no matter which bid the focal bidder
    had chosen that round. Many repeated-auction settings make this
    highest competing bid observable after the fact (e.g. a posted
    clearing/winning price, common in ad-exchange and public-auction
    records). Under that assumption the bidder can, after each round,
    evaluate the reward R(b) = (v0 - b) * 1{b > M_t} it *would have*
    earned for every candidate bid b on its grid -- not just the one it
    actually played.

    This turns each round into an exact single-sample estimate of the
    softmax policy gradient of expected utility (rather than noisy
    single-arm bandit feedback), which converges dramatically faster --
    important here because with n=50 rivals the win probability at any
    given bid is small, so pure bandit feedback would need very many
    rounds to accumulate any learning signal at all.
    """

    def __init__(self, bid_grid: np.ndarray, alpha: float = 150.0):
        self.bid_grid = bid_grid
        self.alpha = alpha
        self.preferences = np.zeros_like(bid_grid)

    def policy(self) -> np.ndarray:
        h = self.preferences - self.preferences.max()
        exp_h = np.exp(h)
        return exp_h / exp_h.sum()

    def act(self, rng: np.random.Generator) -> int:
        probs = self.policy()
        return int(rng.choice(len(self.bid_grid), p=probs))

    def update(self, highest_rival_bid: float, v0: float) -> None:
        """
        Exact softmax policy-gradient step given the realized highest
        rival bid this round: rewards(b) = (v0 - b) * 1{b > M_t} for
        every b on the grid, and
            d/dH(a) E[R] = pi(a) * (R(a) - sum_a' pi(a') R(a'))

        Rewards are expressed as a fraction of v0 so that `alpha` has a
        consistent, dimensionless meaning regardless of the scale of v0
        (an auction with a small v0 and one with a large v0 that are
        otherwise identical in relative terms should learn at the same
        rate).
        """
        rewards = np.where(self.bid_grid > highest_rival_bid, (v0 - self.bid_grid) / v0, 0.0)
        probs = self.policy()
        baseline = np.dot(probs, rewards)
        self.preferences += self.alpha * probs * (rewards - baseline)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    n: int = 50
    v0: float = 0.9
    low: float = 0.0
    high: float = 1.0
    dist_name: str = "uniform"
    dist_kwargs: dict = field(default_factory=dict)
    rounds: int = 20_000
    num_bids: int = 121
    alpha: float = 150.0
    seed: int = 0


@dataclass
class SimulationResult:
    bid_grid: np.ndarray
    chosen_bid_history: np.ndarray
    expected_bid_history: np.ndarray
    utility_history: np.ndarray
    win_history: np.ndarray
    final_policy: np.ndarray
    theoretical_bid: float
    theoretical_utility: float
    theoretical_win_prob: float
    max_win_prob: float


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed)

    dist = build_distribution(cfg.dist_name, cfg.low, cfg.high, **cfg.dist_kwargs)
    equilibrium = EquilibriumBidder(dist, cfg.n, cfg.low, cfg.high)

    bid_grid = np.linspace(0.0, cfg.v0, cfg.num_bids)
    _, best_idx, theoretical_win_prob = equilibrium.best_response(cfg.v0, bid_grid)
    theoretical_bid = float(equilibrium.bid(np.array([cfg.v0]))[0])
    theoretical_bid = min(theoretical_bid, cfg.v0)  # cap, in case of edge cases
    theoretical_utility = (cfg.v0 - theoretical_bid) * equilibrium.win_prob(theoretical_bid)
    # Probability that a round is even "winnable" -- i.e. that bidding at
    # the very top of the grid (v0) would beat all rivals. Learning signal
    # only arrives on such rounds, so this bounds how many rounds are needed.
    max_win_prob = equilibrium.win_prob(cfg.v0)

    learner = PolicyGradientBidder(bid_grid, alpha=cfg.alpha)

    chosen_bid_history = np.empty(cfg.rounds)
    expected_bid_history = np.empty(cfg.rounds)
    utility_history = np.empty(cfg.rounds)
    win_history = np.empty(cfg.rounds, dtype=bool)

    n_rivals = cfg.n - 1

    for t in range(cfg.rounds):
        expected_bid_history[t] = float(np.dot(learner.policy(), bid_grid))

        action_idx = learner.act(rng)
        bid = bid_grid[action_idx]
        chosen_bid_history[t] = bid

        rival_values = np.asarray(dist.rvs(size=n_rivals, random_state=rng))
        rival_bids = equilibrium.bid(rival_values)
        highest_rival_bid = rival_bids.max()

        won = bid > highest_rival_bid
        reward = (cfg.v0 - bid) if won else 0.0

        utility_history[t] = reward
        win_history[t] = won

        learner.update(highest_rival_bid, cfg.v0)

    return SimulationResult(
        bid_grid=bid_grid,
        chosen_bid_history=chosen_bid_history,
        expected_bid_history=expected_bid_history,
        utility_history=utility_history,
        win_history=win_history,
        final_policy=learner.policy(),
        max_win_prob=max_win_prob,
        theoretical_bid=theoretical_bid,
        theoretical_utility=theoretical_utility,
        theoretical_win_prob=theoretical_win_prob,
    )


# ---------------------------------------------------------------------------
# Reporting / plotting
# ---------------------------------------------------------------------------

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def summarize(cfg: SimulationConfig, res: SimulationResult) -> None:
    tail = max(1, cfg.rounds // 20)  # last 5% of rounds
    learned_bid = res.expected_bid_history[-1]
    empirical_bid_mean = res.chosen_bid_history[-tail:].mean()
    empirical_utility_mean = res.utility_history[-tail:].mean()
    empirical_win_rate = res.win_history[-tail:].mean()

    print("=" * 64)
    print("First-Price Sealed-Bid Auction | IPV, symmetric, risk-neutral")
    print("=" * 64)
    print(f"Bidders (n): {cfg.n}")
    print(f"Focal bidder value (v0): {cfg.v0}")
    print(f"Opponent value distribution: {cfg.dist_name} on [{cfg.low}, {cfg.high}] {cfg.dist_kwargs}")
    print(f"Rounds: {cfg.rounds}")
    print("-" * 64)
    print(f"Theoretical equilibrium bid   b*(v0):        {res.theoretical_bid:.4f}")
    print(f"Theoretical win probability:                 {res.theoretical_win_prob:.6g}")
    print(f"Theoretical expected utility:                {res.theoretical_utility:.6g}")
    expected_informative_rounds = cfg.rounds * res.max_win_prob
    print(f"P(round is winnable at all, i.e. bid=v0):    {res.max_win_prob:.6g}")
    print(f"Expected # of 'winnable' rounds simulated:   {expected_informative_rounds:.1f}")
    if expected_informative_rounds < 200:
        print("WARNING: very few winnable rounds occurred -- the learner may not")
        print("         have converged. Increase --rounds or raise --v0 closer to")
        print("         the top of the value distribution's support.")
    print("-" * 64)
    print(f"Learned policy's expected bid (final):       {learned_bid:.4f}")
    print(f"Empirical mean bid (last {tail} rounds):      {empirical_bid_mean:.4f}")
    print(f"Empirical mean utility (last {tail} rounds):  {empirical_utility_mean:.4f}")
    print(f"Empirical win rate (last {tail} rounds):      {empirical_win_rate:.4f}")
    print("=" * 64)


def plot_results(cfg: SimulationConfig, res: SimulationResult) -> None:
    window = max(1, cfg.rounds // 200)
    rounds_axis = np.arange(cfg.rounds)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Learning to Bid in a First-Price Sealed-Bid IPV Auction (n=%d)" % cfg.n)

    # 1. Expected bid under the learned policy vs. theoretical equilibrium bid
    ax = axes[0, 0]
    ax.plot(rounds_axis, res.expected_bid_history, label="Learner's expected bid E[b]", lw=1)
    ax.axhline(res.theoretical_bid, color="black", ls="--", label="Theoretical optimal bid b*(v0)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Bid")
    ax.set_title("Convergence of bidding strategy")
    ax.legend()

    # 2. Moving-average utility vs theoretical optimum
    ax = axes[0, 1]
    ma_utility = moving_average(res.utility_history, window)
    ax.plot(rounds_axis[: len(ma_utility)], ma_utility, label="Moving-avg realized utility", lw=1)
    ax.axhline(res.theoretical_utility, color="black", ls="--", label="Theoretical optimal utility")
    ax.set_xlabel("Round")
    ax.set_ylabel("Utility")
    ax.set_title("Utility over time")
    ax.legend()

    # 3. Moving-average win rate
    ax = axes[1, 0]
    ma_win = moving_average(res.win_history.astype(float), window)
    ax.plot(rounds_axis[: len(ma_win)], ma_win, label="Moving-avg win rate", lw=1)
    ax.axhline(res.theoretical_win_prob, color="black", ls="--", label="Theoretical win probability")
    ax.set_xlabel("Round")
    ax.set_ylabel("Win rate")
    ax.set_title("Win rate over time")
    ax.legend()

    # 4. Final learned policy over the bid grid
    ax = axes[1, 1]
    ax.bar(res.bid_grid, res.final_policy, width=(res.bid_grid[1] - res.bid_grid[0]) * 0.9)
    ax.axvline(res.theoretical_bid, color="black", ls="--", label="Theoretical optimal bid")
    ax.set_xlabel("Bid")
    ax.set_ylabel("Probability")
    ax.set_title("Final learned bidding policy")
    ax.legend()

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50, help="Total number of bidders")
    parser.add_argument("--v0", type=float, default=0.9, help="Fixed value of the focal (learning) bidder")
    parser.add_argument("--low", type=float, default=0.0, help="Lower bound of value distribution support")
    parser.add_argument("--high", type=float, default=1.0, help="Upper bound of value distribution support")
    parser.add_argument("--dist", dest="dist_name", default="uniform",
                         choices=["uniform", "beta", "truncnorm"], help="Opponent value distribution")
    parser.add_argument("--beta-a", type=float, default=2.0, help="Beta distribution 'a' shape parameter")
    parser.add_argument("--beta-b", type=float, default=2.0, help="Beta distribution 'b' shape parameter")
    parser.add_argument("--norm-mu", type=float, default=None, help="Truncated-normal mean")
    parser.add_argument("--norm-sigma", type=float, default=None, help="Truncated-normal std dev")
    parser.add_argument("--rounds", type=int, default=20_000, help="Number of auction rounds to simulate")
    parser.add_argument("--num-bids", type=int, default=121, help="Number of discretized candidate bids")
    parser.add_argument("--alpha", type=float, default=150.0, help="Policy-gradient learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting; only print the summary")
    args = parser.parse_args()

    dist_kwargs = {}
    if args.dist_name == "beta":
        dist_kwargs = {"beta_a": args.beta_a, "beta_b": args.beta_b}
    elif args.dist_name == "truncnorm":
        dist_kwargs = {}
        if args.norm_mu is not None:
            dist_kwargs["norm_mu"] = args.norm_mu
        if args.norm_sigma is not None:
            dist_kwargs["norm_sigma"] = args.norm_sigma

    cfg = SimulationConfig(
        n=args.n,
        v0=args.v0,
        low=args.low,
        high=args.high,
        dist_name=args.dist_name,
        dist_kwargs=dist_kwargs,
        rounds=args.rounds,
        num_bids=args.num_bids,
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
