"""Public-API stability — pin the exported surface and its signatures.

If a test here breaks, the public contract changed: update it deliberately (and
bump the version), don't just edit the assertion away.
"""

from __future__ import annotations

import inspect
import importlib
import dataclasses

PKG = importlib.import_module("mobile_robot_navigation")


def test_exports() -> None:
    """`__all__` is exactly the supported public surface."""
    assert set(PKG.__all__) == {
        "train",
        "TrainResult",
        "Config",
        "Actor",
        "Critic",
        "OUNoise",
        "MobileRobotEnv",
    }


def test_train_signature() -> None:
    """The public entry keeps its (cfg, *, device) -> TrainResult contract."""
    sig = inspect.signature(PKG.train, eval_str=True)
    assert list(sig.parameters) == ["cfg", "device"]
    assert sig.return_annotation is PKG.TrainResult


def test_trainresult_fields() -> None:
    """The TrainResult dataclass keeps its public fields."""
    fields = [f.name for f in dataclasses.fields(PKG.TrainResult)]
    assert fields == ["actor", "episode_rewards"]
