"""Experimental multi-trust-region batch allocation, scoring, and candidate selection."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np

from .region_types import (
    CandidateProposal,
    RegionBatch,
    RegionCandidate,
    SharingPolicy,
)


class RegionRound(NamedTuple):
    """Batches, proposals, scored candidates, and selected arms from one search round."""

    batches: list[RegionBatch]
    proposals: list[CandidateProposal]
    candidates: list[RegionCandidate]
    selected: list[RegionCandidate]


def _policy(sharing_policy: SharingPolicy | str) -> str:
    if isinstance(sharing_policy, SharingPolicy):
        return sharing_policy.value
    return str(sharing_policy)


def make_region(
    num_dim: int,
    num_regions: int = 4,
    sharing_policy: SharingPolicy | str = SharingPolicy.BOX_CONTAIN,
    seed: int | None = None,
):
    """Construct a native `MultiTrustRegion` state with the given sharing policy."""
    from . import MultiTrustRegion

    return MultiTrustRegion(
        num_dim=num_dim,
        num_regions=num_regions,
        sharing_policy=_policy(sharing_policy),
        seed=seed,
    )


def allocate_batches(
    state,
    budget: int,
    utility: Sequence[float] | np.ndarray | None = None,
) -> list[RegionBatch]:
    """Allocate candidate generation slices across active trust regions."""
    raw_batches = (
        state.allocate(budget)
        if utility is None
        else state.allocate_with(budget, np.asarray(utility, dtype=float))
    )
    return [RegionBatch(*batch) for batch in raw_batches]


def select_candidates(
    state,
    candidates: Sequence[RegionCandidate | Sequence[Any]],
    num_arms: int,
) -> list[RegionCandidate]:
    """Select the top `num_arms` candidates across regions using `state`."""
    raw_candidates = [
        candidate
        if isinstance(candidate, RegionCandidate)
        else RegionCandidate(*candidate)
        for candidate in candidates
    ]
    res = state.select(
        [
            (candidate.index, candidate.region, candidate.seed, candidate.score)
            for candidate in raw_candidates
        ],
        num_arms,
    )
    if isinstance(res, tuple) and len(res) == 2:
        selected, _ = res
    else:
        selected = res
    return [RegionCandidate(*candidate) for candidate in selected]


def _coerceproposal(
    candidate: CandidateProposal | Sequence[Any],
) -> CandidateProposal:
    if isinstance(candidate, CandidateProposal):
        return candidate
    region, seed, payload = candidate
    return CandidateProposal(int(region), int(seed), payload)


class RegionalSearchLoop:
    """End-to-end experimental loop around a multi-trust-region state."""

    def __init__(self, state) -> None:
        self._inner = state

    @property
    def state(self):
        """Return the underlying immutable Rust MultiTrustRegion state."""
        return self._inner

    @state.setter
    def state(self, value):
        self._inner = value

    def allocate(
        self, budget: int, utility: Sequence[float] | np.ndarray | None = None
    ) -> list[RegionBatch]:
        """Allocate candidate generation batches across regions for `budget`."""
        return allocate_batches(self._inner, budget, utility)

    def propose(
        self,
        budget: int,
        proposal_fn: Callable[
            [RegionBatch],
            Sequence[CandidateProposal | Sequence[Any]] | CandidateProposal | None,
        ],
        utility: Sequence[float] | np.ndarray | None = None,
    ) -> tuple[list[RegionBatch], list[CandidateProposal]]:
        """Allocate region batches and collect candidate proposals from `proposal_fn`."""
        batches = self.allocate(budget, utility)
        proposals: list[CandidateProposal] = []
        for batch in batches:
            produced = proposal_fn(batch)
            if produced is None:
                continue
            if isinstance(produced, CandidateProposal):
                proposals.append(produced)
                continue
            proposals.extend(map(_coerceproposal, produced))
        return batches, proposals

    def score(
        self,
        proposals: Sequence[CandidateProposal | Sequence[Any]],
        scorer: Callable[[CandidateProposal], float],
    ) -> list[RegionCandidate]:
        """Evaluate `scorer` on each proposal and return indexed `RegionCandidate` records."""
        scored: list[RegionCandidate] = []
        for index, proposal in enumerate(map(_coerceproposal, proposals)):
            scored.append(
                RegionCandidate(
                    index=index,
                    region=proposal.region,
                    seed=proposal.seed,
                    score=float(scorer(proposal)),
                )
            )
        return scored

    def select(
        self, candidates: Sequence[RegionCandidate | Sequence[Any]], num_arms: int
    ) -> list[RegionCandidate]:
        """Select `num_arms` candidates from `candidates` using the underlying state."""
        raw_candidates = [
            candidate
            if isinstance(candidate, RegionCandidate)
            else RegionCandidate(*candidate)
            for candidate in candidates
        ]
        res = self._inner.select(
            [
                (candidate.index, candidate.region, candidate.seed, candidate.score)
                for candidate in raw_candidates
            ],
            num_arms,
        )
        if isinstance(res, tuple) and len(res) == 2:
            selected, self._inner = res
        else:
            selected = res
        return [RegionCandidate(*candidate) for candidate in selected]

    def run_round(
        self,
        *,
        budget: int,
        proposal_fn: Callable[
            [RegionBatch],
            Sequence[CandidateProposal | Sequence[Any]] | CandidateProposal | None,
        ],
        scorer: Callable[[CandidateProposal], float],
        num_arms: int | None = None,
        utility: Sequence[float] | np.ndarray | None = None,
    ) -> RegionRound:
        """Execute one full propose-score-select round and return a `RegionRound`."""
        batches, proposals = self.propose(budget, proposal_fn, utility)
        candidates = self.score(proposals, scorer)
        selected = self.select(candidates, num_arms or len(candidates))
        return RegionRound(
            batches=batches,
            proposals=proposals,
            candidates=candidates,
            selected=selected,
        )

    def tell(self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None):
        """Record observations `(x, y, y_var)` into the multi-trust-region state."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x.flags.writeable = False
        y.flags.writeable = False
        if y_var is not None:
            y_var = np.asarray(y_var, dtype=float)
            y_var.flags.writeable = False
        res = self._inner.tell(x, y) if y_var is None else self._inner.tell(x, y, y_var)
        if res is not None:
            self._inner = res
        return self._inner

    def restart_region(self, region: int, new_center: np.ndarray):
        """Restart `region` centered at `new_center`."""
        res = self._inner.restart_region(region, new_center)
        if res is not None:
            self._inner = res
        return self._inner

    def variance(self, region: int):
        """Return the current posterior variance estimate for `region`."""
        return self._inner.variance(region)

    def fork(self) -> RegionalSearchLoop:
        """Fork the search loop with an independent copy of the underlying state."""
        inner = self._inner.fork() if hasattr(self._inner, "fork") else self._inner
        return RegionalSearchLoop(inner)


multi_region = sys.modules[__name__]


__all__ = [
    "CandidateProposal",
    "RegionBatch",
    "RegionCandidate",
    "RegionRound",
    "RegionalSearchLoop",
    "SharingPolicy",
    "allocate_batches",
    "make_region",
    "multi_region",
    "select_candidates",
]
