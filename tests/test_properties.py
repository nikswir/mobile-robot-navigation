"""Property-based tests for the DDPG code (Hypothesis, CPU, stage 1).

Assert real invariants over generated inputs rather than a few hand-picked
cases: the actor's tanh bound, the noise process's reset / sampling contract,
and the environment's fixed-length observation over randomized layouts.
"""

from __future__ import annotations

import math

import torch
import numpy as np

from hypothesis import given, settings
from hypothesis import strategies as st

from mobile_robot_navigation.agent import Actor, OUNoise
from mobile_robot_navigation.environment import ChopperScape

# A single CPU device keeps the tests deterministic and GPU-independent.
CPU = torch.device("cpu")

# Finite, modestly-bounded floats: enough to exercise the maths without
# overflowing intermediate activations into non-finite territory.
FLOATS = st.floats(
    min_value=-1e3,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


########################################
#               Networks               #
########################################


@settings(max_examples=40)
@given(state=st.lists(FLOATS, min_size=10, max_size=10))
def test_actor_output_is_within_tanh_bounds(state: list[float]) -> None:
    """The actor's tanh head keeps every action in [-1, 1] and finite."""
    torch.manual_seed(0)
    actor = Actor(10, 2).to(CPU)

    x = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        out = actor(x)

    assert torch.isfinite(out).all()
    assert bool((out >= -1.0).all())
    assert bool((out <= 1.0).all())


########################################
#          Exploration noise           #
########################################


@settings(max_examples=40)
@given(
    action_size=st.integers(min_value=1, max_value=5),
    mu=FLOATS,
)
def test_ou_noise_reset_and_sample(action_size: int, mu: float) -> None:
    """`reset()` restores the state to mu; `sample()` is finite & sized."""
    noise = OUNoise(action_size, mu=mu, device=CPU)

    # ── reset() puts the state back at the mean ──
    noise.state = np.zeros(action_size) + 123.0
    noise.reset()
    assert np.allclose(noise.state, np.ones(action_size) * mu)

    # ── sample() returns a finite tensor of length action_size ──
    sample = noise.sample()
    assert sample.shape == (action_size,)
    assert torch.isfinite(sample).all()


########################################
#             Environment              #
########################################


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**16))
def test_reset_returns_bounded_length_10_observation(seed: int) -> None:
    """A reset (fresh random layout) yields 10 bounded observation values."""
    env = ChopperScape(seed=seed)

    obs = env.reset()

    assert len(obs) == 10
    assert all(math.isfinite(v) for v in obs)

    # ── Seven normalised scans plus angle ratios fall in [0, 1] ──
    for value in obs[:9]:
        assert 0.0 <= value <= 1.0

    # ── Normalised distance to the active POI stays non-negative ──
    assert obs[9] >= 0.0
