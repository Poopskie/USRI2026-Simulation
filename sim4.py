"""
First-Price Sealed-Bid Auction -- Reserve Prices + Verifiable Types
=======================================================================

Builds on sim3.py's two-round qualification mechanism by adding two more
rules that push this well outside any known theoretical result. We are in
genuinely uncharted territory here -- treat everything below as an
exploratory estimate, not a validated equilibrium.

Recap of sim3's mechanism
---------------------------
1. Round 1 (screening, NON-BINDING): every one of the N participating
   bidders submits a sealed bid, used only to rank entrants.
2. Only the top `n_advance` bidders (by round-1 bid) advance to round 2.
3. Round 2 (binding): advancing bidders submit a second sealed bid, which
   cannot be below their own round-1 bid. Highest round-2 bid among the
   qualifiers wins and pays that bid.

New rule 1: a learning seller sets a reserve price after round 1
--------------------------------------------------------------------
After round 1 closes and qualifiers are determined, the seller observes
the top qualifying round-1 bid (the most informative signal available)
and sets a reserve price for round 2 -- bidders know the seller will do
this. A round-2 bid only wins (and only then does a sale happen) if it
beats both the other qualifier(s) AND the reserve. The seller learns its
reserve-setting policy the same way every other agent in this file
learns: a softmax table indexed by the observed signal, updated from
realized revenue.

Hypothesis (the classic dynamic-mechanism-design "ratchet effect",
Laffont & Tirole 1988): bidders will realize that bidding high in round 1
invites a higher reserve in round 2, so they have an incentive to shade
down their round-1 bid to look weaker than they are. If this dominates,
round-1 bids become uninformative and the seller's learned reserve should
collapse toward 0. We test this in isolation first (`--no-reserve` /
`--no-message-constraint` flags let you turn either new rule off) before
layering on rule 2.

New rule 2: verifiable type => a minimum-bid message constraint
--------------------------------------------------------------------
The seller can verify whether a bidder is strong or weak (e.g. some
observable credential), even though it can't see the bidder's exact
value. This is modeled as a hard constraint on the bid message space: a
bidder's bid can never go below their OWN type's distribution floor.
Weak bidders (floor 0) are unaffected; strong bidders (floor 0.5) can no
longer shade a round-1 bid all the way down to look weak -- they are
stuck signaling "at least strong-tier" no matter what. This partially
blunts the ratchet effect for strong bidders specifically, since they
can't pool with weak bidders at very low bids, and is the alternative
modeling choice worth comparing against a coarser "few discrete message
categories" constraint if this one doesn't produce interesting dynamics.

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

- The seller (`ReserveSeller`) uses the same single-action
  REINFORCE-with-baseline update as round 1: it observes a discretized
  "signal" bin (the top qualifying round-1 bid), picks a reserve from a
  grid, and updates toward reserves that realize higher revenue.

Usage
-----
    py sim4.py
    py sim4.py --n-advance 2 --rounds 100000
    py sim4.py --no-reserve                     # isolate the message constraint
    py sim4.py --no-message-constraint          # isolate the reserve/ratchet effect
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
        n_min = kwargs.get("n_min", 2)
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
                 num_value_bins: int = 21, alpha: float = 15.0,
                 decay_scale: float = 100.0, pref_decay: float = 1e-4):
        self.low = low
        self.high = high
        self.fraction_grid = fraction_grid
        self.alpha0 = alpha
        self.num_value_bins = num_value_bins
        self.bin_edges = np.linspace(low, high, num_value_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.preferences = np.zeros((num_value_bins, len(fraction_grid)))
        # Same anti-premature-saturation machinery as QualificationBidder.
        self.decay_scale = decay_scale
        self.pref_decay = pref_decay
        self.visit_count = np.zeros(num_value_bins)

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
        self.preferences *= (1.0 - self.pref_decay)

        surplus = np.maximum(values - floors, 1e-12)
        candidate_bids = floors[:, None] + self.fraction_grid[None, :] * surplus[:, None]
        rewards = np.where(candidate_bids > highest_other[:, None],
                            1.0 - self.fraction_grid[None, :], 0.0)
        baseline = np.sum(probs_matrix * rewards, axis=1, keepdims=True)

        effective_alpha = self.alpha0 / (1.0 + self.visit_count[bin_idxs] / self.decay_scale)
        delta = effective_alpha[:, None] * probs_matrix * (rewards - baseline)
        np.add.at(self.preferences, bin_idxs, delta)
        np.add.at(self.visit_count, bin_idxs, 1.0)

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
                 num_value_bins: int = 21, alpha: float = 15.0, baseline_decay: float = 0.02,
                 message_floor: float = 0.0, decay_scale: float = 100.0, pref_decay: float = 1e-4):
        self.low = low
        self.high = high
        self.fraction_grid = fraction_grid
        self.alpha0 = alpha
        self.baseline_decay = baseline_decay
        self.num_value_bins = num_value_bins
        self.bin_edges = np.linspace(low, high, num_value_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.preferences = np.zeros((num_value_bins, len(fraction_grid)))
        self.baseline = np.zeros(num_value_bins)
        self._baseline_init = np.zeros(num_value_bins, dtype=bool)
        # Message-space / verifiability constraint: this bidder can never
        # report a bid below `message_floor` (its own type's distribution
        # floor, when the constraint is enabled -- 0 otherwise, i.e. the
        # unconstrained sim3 behavior).
        self.message_floor = message_floor
        # Anti-premature-saturation machinery (see class docstring update
        # rule): a per-bin visit count that shrinks the effective learning
        # rate as a bin accumulates evidence (Robbins-Monro-style decay),
        # plus a tiny continuous pull of every preference back toward 0
        # each update so the softmax can never fully saturate to a
        # zero-entropy, permanently-frozen policy.
        self.decay_scale = decay_scale
        self.pref_decay = pref_decay
        self.visit_count = np.zeros(num_value_bins)

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

        surplus = np.maximum(values - self.message_floor, 0.0)
        bids = self.message_floor + self.fraction_grid[frac_idxs] * surplus
        return bids, bin_idxs, frac_idxs, probs_matrix

    def update_batch(self, bin_idxs: np.ndarray, frac_idxs: np.ndarray,
                      rewards: np.ndarray, probs_matrix: np.ndarray) -> None:
        # Gentle continuous regularization: pull every preference a tiny
        # bit back toward 0 (uniform policy) each round, so a bin can never
        # fully saturate to zero-entropy and get permanently stuck.
        self.preferences *= (1.0 - self.pref_decay)

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

        # Robbins-Monro-style decaying step size: bins with more accumulated
        # visits take smaller steps, so the policy settles rather than
        # perpetually reacting to noise -- but never fully stops adapting.
        effective_alpha = self.alpha0 / (1.0 + self.visit_count[bin_idxs] / self.decay_scale)
        delta = effective_alpha[:, None] * advantage[:, None] * (indicator - probs_matrix)
        np.add.at(self.preferences, bin_idxs, delta)
        np.add.at(self.visit_count, bin_idxs, 1.0)

    def expected_bid_function(self) -> np.ndarray:
        probs = np.array([self.policy_row(i) for i in range(self.num_value_bins)])
        expected_fraction = probs @ self.fraction_grid
        surplus = np.maximum(self.bin_centers - self.message_floor, 0.0)
        return self.message_floor + expected_fraction * surplus


# ---------------------------------------------------------------------------
# Learning seller: sets a round-2 reserve price from the round-1 signal
# ---------------------------------------------------------------------------

class ReserveSeller:
    """
    Learns a reserve-setting policy: after observing a discretized SIGNAL
    (the top qualifying round-1 bid), pick a MARKUP fraction m in [0, 1]
    from a grid, and set reserve = signal + m * (high - signal).

    Why parameterize it this way rather than an absolute reserve level:
    round-2 bids are already floored at each qualifier's own round-1 bid,
    so in particular the top qualifier (whose floor equals the signal)
    will never bid below the signal anyway. A reserve <= signal can only
    ever bind on a WEAKER qualifier, never on the top one -- it's a
    strictly weaker tool than one that's guaranteed to sit at or above
    what the round-1 floor already guarantees. Forcing reserve >= signal
    makes the reserve a genuine additional extraction lever on top of the
    qualification floor, rather than a partially redundant one.

    Updated via the same single-action REINFORCE-with-baseline rule as
    `QualificationBidder`, fed the REALIZED round-2 revenue (0 if no sale,
    e.g. because the winning bid didn't clear the reserve). We
    deliberately reuse the *simple* bandit-style update here (rather than
    a full-information lookahead over every candidate reserve) to keep
    this first pass quick, consistent with the rest of sim3/sim4.
    """

    def __init__(self, high: float, markup_grid: np.ndarray, num_signal_bins: int = 11,
                 alpha: float = 15.0, baseline_decay: float = 0.02,
                 decay_scale: float = 100.0, pref_decay: float = 1e-4):
        self.high = high
        self.markup_grid = markup_grid
        self.alpha0 = alpha
        self.baseline_decay = baseline_decay
        self.num_signal_bins = num_signal_bins
        self.bin_edges = np.linspace(0.0, high, num_signal_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.preferences = np.zeros((num_signal_bins, len(markup_grid)))
        self.baseline = np.zeros(num_signal_bins)
        self._baseline_init = np.zeros(num_signal_bins, dtype=bool)
        # Same anti-premature-saturation machinery as QualificationBidder.
        self.decay_scale = decay_scale
        self.pref_decay = pref_decay
        self.visit_count = np.zeros(num_signal_bins)

    def bin_index(self, signal: float) -> int:
        idx = np.searchsorted(self.bin_edges, signal, side="right") - 1
        return int(np.clip(idx, 0, self.num_signal_bins - 1))

    def policy_row(self, bin_idx: int) -> np.ndarray:
        h = self.preferences[bin_idx]
        h = h - h.max()
        exp_h = np.exp(h)
        return exp_h / exp_h.sum()

    def act(self, signal: float, rng: np.random.Generator) -> tuple[float, int, int, np.ndarray]:
        bin_idx = self.bin_index(signal)
        probs = self.policy_row(bin_idx)
        markup_idx = int(rng.choice(len(self.markup_grid), p=probs))
        reserve = signal + self.markup_grid[markup_idx] * (self.high - signal)
        return reserve, bin_idx, markup_idx, probs

    def update(self, bin_idx: int, markup_idx: int, revenue_reward: float, probs: np.ndarray) -> None:
        self.preferences *= (1.0 - self.pref_decay)

        if not self._baseline_init[bin_idx]:
            self.baseline[bin_idx] = revenue_reward
            self._baseline_init[bin_idx] = True
        else:
            self.baseline[bin_idx] += self.baseline_decay * (revenue_reward - self.baseline[bin_idx])

        indicator = np.zeros_like(probs)
        indicator[markup_idx] = 1.0
        advantage = revenue_reward - self.baseline[bin_idx]
        effective_alpha = self.alpha0 / (1.0 + self.visit_count[bin_idx] / self.decay_scale)
        self.preferences[bin_idx] += effective_alpha * advantage * (indicator - probs)
        self.visit_count[bin_idx] += 1.0

    def expected_reserve_function(self) -> np.ndarray:
        """Illustrative reserve(signal), using each bin's own center as the representative signal."""
        probs = np.array([self.policy_row(i) for i in range(self.num_signal_bins)])
        expected_markup = probs @ self.markup_grid
        return self.bin_centers + expected_markup * (self.high - self.bin_centers)


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
    alpha: float = 15.0
    seed: int = 0
    enable_reserve: bool = True
    enable_message_constraint: bool = True
    num_signal_bins: int = 11
    num_reserve_actions: int = 41
    decay_scale: float = 100.0
    pref_decay: float = 1e-4


@dataclass
class SimulationResult:
    fraction_grid: np.ndarray
    strong_round1: QualificationBidder
    weak_round1: QualificationBidder
    strong_round2: TopUpBidder
    weak_round2: TopUpBidder
    seller: "ReserveSeller | None"
    strong_ref_bid1_history: np.ndarray
    weak_ref_bid1_history: np.ndarray
    qualify_rate_history: dict  # {"strong": array, "weak": array}
    win_rate_history: dict      # {"strong": array, "weak": array}
    reserve_ref_history: np.ndarray
    sale_rate_history: np.ndarray
    revenue_history: np.ndarray
    mean_n_history: np.ndarray


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed)

    n_dist = build_n_distribution(cfg.n_dist_name, **cfg.n_dist_kwargs)
    strong_value_dist = build_distribution(cfg.strong.dist_name, cfg.strong.low, cfg.strong.high, **cfg.strong.dist_kwargs)
    weak_value_dist = build_distribution(cfg.weak.dist_name, cfg.weak.low, cfg.weak.high, **cfg.weak.dist_kwargs)

    fraction_grid = np.linspace(1.0 / cfg.num_actions, 1.0, cfg.num_actions)

    strong_msg_floor = cfg.strong.low if cfg.enable_message_constraint else 0.0
    weak_msg_floor = cfg.weak.low if cfg.enable_message_constraint else 0.0
    strong_round1 = QualificationBidder(cfg.strong.low, cfg.strong.high, fraction_grid, cfg.num_value_bins,
                                         cfg.alpha, message_floor=strong_msg_floor,
                                         decay_scale=cfg.decay_scale, pref_decay=cfg.pref_decay)
    weak_round1 = QualificationBidder(cfg.weak.low, cfg.weak.high, fraction_grid, cfg.num_value_bins,
                                       cfg.alpha, message_floor=weak_msg_floor,
                                       decay_scale=cfg.decay_scale, pref_decay=cfg.pref_decay)
    strong_round2 = TopUpBidder(cfg.strong.low, cfg.strong.high, fraction_grid, cfg.num_value_bins, cfg.alpha,
                                 decay_scale=cfg.decay_scale, pref_decay=cfg.pref_decay)
    weak_round2 = TopUpBidder(cfg.weak.low, cfg.weak.high, fraction_grid, cfg.num_value_bins, cfg.alpha,
                               decay_scale=cfg.decay_scale, pref_decay=cfg.pref_decay)

    high = cfg.strong.high
    seller = None
    if cfg.enable_reserve:
        markup_grid = np.linspace(0.0, 1.0, cfg.num_reserve_actions)
        seller = ReserveSeller(high, markup_grid, cfg.num_signal_bins, cfg.alpha,
                                decay_scale=cfg.decay_scale, pref_decay=cfg.pref_decay)
    signal_ref = 0.7 * high

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
    reserve_ref_history = np.empty(cfg.rounds)
    sale_rate_history = np.empty(cfg.rounds)
    revenue_history = np.empty(cfg.rounds)
    mean_n_history = np.empty(cfg.rounds)

    p_strong = cfg.strong.prob / (cfg.strong.prob + cfg.weak.prob)

    for t in range(cfg.rounds):
        strong_ref_bid1_history[t] = strong_msg_floor + strong_round1.policy_row(strong_ref_bin) @ fraction_grid * max(strong_ref_value - strong_msg_floor, 0.0)
        weak_ref_bid1_history[t] = weak_msg_floor + weak_round1.policy_row(weak_ref_bin) @ fraction_grid * max(weak_ref_value - weak_msg_floor, 0.0)
        if seller is not None:
            expected_markup_ref = seller.policy_row(seller.bin_index(signal_ref)) @ seller.markup_grid
            reserve_ref_history[t] = signal_ref + expected_markup_ref * (high - signal_ref)
        else:
            reserve_ref_history[t] = 0.0

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

        # ---- Seller: observe signal, set reserve for round 2 ----
        signal = bids1[qualifier_idx].max()
        if seller is not None:
            reserve, sig_bin, res_idx, sell_probs = seller.act(signal, rng)
        else:
            reserve, sig_bin, res_idx, sell_probs = 0.0, None, None, None

        # ---- Round 2: binding, bid >= own round-1 bid, among qualifiers only ----
        payoff = np.zeros(n_total)  # realized downstream utility, used as round-1 reward
        winner_among_qualifiers = None
        sale_happened = False
        revenue = 0.0

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

            # The reserve is an EXTRA bar a bid must clear to win, on top of
            # beating the other qualifier(s) -- fold it into the same
            # "threshold to beat" used by TopUpBidder's full-info update.
            effective_threshold = np.maximum(highest_other2, reserve)

            sale_happened = bool(q_bids2[winner_local] >= reserve)
            revenue = float(q_bids2[winner_local]) if sale_happened else 0.0

            q_reward = np.where((np.arange(n_advance_actual) == winner_local) & sale_happened,
                                 q_values - q_bids2, 0.0)

            if len(q_strong_pos):
                probs2_strong = np.array([q_probs2[p] for p in q_strong_pos])
                strong_round2.update_batch(q_bin2[q_strong_pos], q_values[q_strong_pos], q_floors[q_strong_pos],
                                            effective_threshold[q_strong_pos], probs2_strong)
            if len(q_weak_pos):
                probs2_weak = np.array([q_probs2[p] for p in q_weak_pos])
                weak_round2.update_batch(q_bin2[q_weak_pos], q_values[q_weak_pos], q_floors[q_weak_pos],
                                          effective_threshold[q_weak_pos], probs2_weak)

            payoff[qualifier_idx] = q_reward
            if sale_happened:
                winner_among_qualifiers = qualifier_idx[winner_local]

        if seller is not None:
            seller.update(sig_bin, res_idx, revenue / high, sell_probs)

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
        sale_rate_history[t] = 1.0 if sale_happened else 0.0
        revenue_history[t] = revenue

    return SimulationResult(
        fraction_grid=fraction_grid,
        strong_round1=strong_round1,
        weak_round1=weak_round1,
        strong_round2=strong_round2,
        weak_round2=weak_round2,
        seller=seller,
        strong_ref_bid1_history=strong_ref_bid1_history,
        weak_ref_bid1_history=weak_ref_bid1_history,
        qualify_rate_history={"strong": strong_qualify_history, "weak": weak_qualify_history},
        win_rate_history={"strong": strong_win_history, "weak": weak_win_history},
        reserve_ref_history=reserve_ref_history,
        sale_rate_history=sale_rate_history,
        revenue_history=revenue_history,
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
    print("First-Price Sealed-Bid Auction | Reserve Prices + Verifiable Types")
    print("=" * 70)
    print("No known equilibrium for this mechanism -- numbers below are a")
    print("SIMULATION-BASED ESTIMATE from self-play, and round-1 / the seller's")
    print("reserve policy in particular may not have converged (see docstring).")
    print("-" * 70)
    print(f"Strong type: value ~ {cfg.strong.dist_name} on [{cfg.strong.low}, {cfg.strong.high}], "
          f"P(strong) = {cfg.strong.prob / (cfg.strong.prob + cfg.weak.prob):.2f}")
    print(f"Weak type:   value ~ {cfg.weak.dist_name} on [{cfg.weak.low}, {cfg.weak.high}], "
          f"P(weak)   = {cfg.weak.prob / (cfg.strong.prob + cfg.weak.prob):.2f}")
    print(f"Number-of-bidders distribution: {cfg.n_dist_name} {cfg.n_dist_kwargs}")
    print(f"n_advance (qualifiers to round 2): {cfg.n_advance}")
    print(f"Reserve mechanism enabled:    {cfg.enable_reserve}")
    print(f"Message-floor constraint enabled: {cfg.enable_message_constraint}")
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
    print("-" * 70)
    print(f"Seller's reserve at signal=0.7*high (final): {res.reserve_ref_history[-1]:.4f}")
    print(f"Sale rate (last {tail} rounds):               {res.sale_rate_history[-tail:].mean():.4f}")
    print(f"Mean revenue (last {tail} rounds):            {res.revenue_history[-tail:].mean():.4f}")
    print("=" * 70)


def plot_results(cfg: SimulationConfig, res: SimulationResult) -> None:
    window = max(1, cfg.rounds // 200)
    rounds_axis = np.arange(cfg.rounds)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Self-Play Estimate: Reserve Prices + Verifiable Types")

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

    ax = axes[0, 2]
    ax.plot(rounds_axis, res.reserve_ref_history, color="tab:red", lw=1,
            label="E[reserve] at signal=0.7*high")
    ax.set_xlabel("Round")
    ax.set_ylabel("Reserve")
    ax.set_title("Seller's reserve-price convergence\n(ratchet-effect check: does this -> 0?)")
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

    ax = axes[1, 2]
    if res.seller is not None:
        reserve_fn = res.seller.expected_reserve_function()
        ax.plot(res.seller.bin_centers, reserve_fn, marker="o", ms=3, color="tab:red",
                 label="Learned reserve(signal)")
        ax.plot([0, 1], [0, 1], ls=":", color="gray", label="reserve = signal")
        ax.set_xlabel("Signal (top qualifying round-1 bid)")
        ax.set_ylabel("Reserve")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Reserve mechanism disabled", ha="center", va="center")
    ax.set_title("Learned reserve-setting policy")

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
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=15)
    parser.add_argument("--n-advance", type=int, default=2, help="Number of bidders who advance to round 2")

    parser.add_argument("--rounds", type=int, default=20_000)
    parser.add_argument("--num-actions", type=int, default=41, help="Resolution of the bid-shading fraction grid")
    parser.add_argument("--num-value-bins", type=int, default=21, help="Resolution of the value discretization per type")
    parser.add_argument("--alpha", type=float, default=15.0, help="Policy-gradient learning rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")

    parser.add_argument("--no-reserve", dest="enable_reserve", action="store_false",
                         help="Disable the seller's round-2 reserve mechanism")
    parser.add_argument("--no-message-constraint", dest="enable_message_constraint", action="store_false",
                         help="Disable the type-verifiability minimum-bid constraint")
    parser.add_argument("--num-signal-bins", type=int, default=11, help="Resolution of the seller's signal discretization")
    parser.add_argument("--num-reserve-actions", type=int, default=41, help="Resolution of the reserve-price grid")
    parser.add_argument("--decay-scale", type=float, default=100.0,
                         help="Visit-count scale for the learning-rate decay (larger = slower decay)")
    parser.add_argument("--pref-decay", type=float, default=1e-4,
                         help="Per-round preference regularization strength (prevents policy saturation)")
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
        enable_reserve=args.enable_reserve,
        enable_message_constraint=args.enable_message_constraint,
        num_signal_bins=args.num_signal_bins,
        num_reserve_actions=args.num_reserve_actions,
        decay_scale=args.decay_scale,
        pref_decay=args.pref_decay,
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
