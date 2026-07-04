# mobile-robot-navigation

A DDPG (Deep Deterministic Policy Gradient) reinforcement-learning agent that
learns to navigate a mobile robot through a 2-D field filled with obstacles
toward a target. Every episode samples a **fresh random layout** — obstacle
count, sizes, positions and the start pose all vary, with a BFS reachability
check guaranteeing the task stays solvable — so the learned policy generalizes
across maps instead of memorizing one. The robot perceives its surroundings
with a lidar-style angular `scan()` and steers exploration with a
point-of-interest (POI) heuristic, while the DDPG actor-critic networks learn
a continuous steering policy from experience.

<p align="center">
  <img src="docs/assets/demo.gif" alt="Trained policy navigating random layouts" width="640">
</p>

*The trained policy navigating previously unseen random layouts.*

## Setup

```bash
uv sync
uv run pre-commit install
```

## Run

```bash
uv run python -m mobile_robot_navigation.run            # run with the default config
uv run python -m mobile_robot_navigation.run --cfg job  # print the composed config
```

## Reproduce the experiment

```bash
uv run python report/scripts/run_experiment.py  # train + save actor & rewards
uv run python report/scripts/eval_policy.py     # arrival rate, 300 layouts
uv run python report/scripts/make_figures.py    # regenerate report figures
uv run python report/scripts/make_gif.py        # rebuild the README demo GIF
```

## Develop

The engineering workflow (toolchain, the pre-commit gate, two-stage tests, CI)
is documented in [AGENTS.md](AGENTS.md). Common tasks: `just lint`, `just test`
— run `just` for the list. In short:

```bash
uv run pre-commit run --all-files   # lint, format, types, style checks
uv run pytest                       # stage-1 (fast, CPU) tests
RUN_STAGE2=1 uv run pytest          # + heavy tests (GPU / training)
```

See [docs/architecture.md](docs/architecture.md) for how a training run flows
through the package.
