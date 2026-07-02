"""DDPG agent: networks, exploration noise and the training loop.

Holds the Actor / Critic networks, the Ornstein-Uhlenbeck exploration noise,
the replay-buffer helpers and `train_policy`, which runs the full DDPG training
loop against a `ChopperScape` environment. Tensors are routed through a single
`DEVICE` chosen once (CUDA, then Apple MPS, otherwise CPU).
"""

from __future__ import annotations

import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import LinearLR

from mobile_robot_navigation.environment import ChopperScape


# Single device chosen once; every tensor is routed through it instead of
# pinning CUDA at the call sites (kept CPU-friendly for tests).
def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _select_device()

########################################
#               Networks               #
########################################


class Actor(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden1: int = 128,
        hidden2: int = 256,
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, self.action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        value = torch.tanh(self.linear3(x))
        return value


class Critic(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden1: int = 128,
        hidden2: int = 256,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2(x))
        value: torch.Tensor = self.linear3(x)
        return value


########################################
#          Exploration noise           #
########################################


class OUNoise:
    """
    Ornstein-Uhlenbeck process для генерации коррелированного шума.
    Позволяет агенту исследовать пространство действий более эффективно,
    чем простой случайный шум.
    """

    def __init__(
        self,
        action_size: int,
        mu: float = 0,
        theta: float = 0.15,
        sigma: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        self.action_size = action_size
        self.mu = mu  # Среднее значение (обычно 0)
        self.theta = theta  # Скорость возврата к среднему (0.1-0.2)
        self.sigma = sigma  # Волатильность шума (0.1-0.3)
        self.device = device if device is not None else DEVICE
        self.state = np.ones(self.action_size) * self.mu
        self.reset()

    def reset(self) -> None:
        """Сбросить состояние шума к среднему"""
        self.state = np.ones(self.action_size) * self.mu

    def sample(self) -> torch.Tensor:
        """Генерация следующего значения шума"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.FloatTensor(self.state).to(self.device)


########################################
#        Weight initialization         #
########################################


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


########################################
#        Target & replay buffer        #
########################################


def update_target_network(
    target_network: nn.Module,
    source_network: nn.Module,
    tau: float,
) -> None:
    pairs = zip(
        target_network.parameters(),
        source_network.parameters(),
        strict=False,
    )
    for target_param, param in pairs:
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau,
        )


def add_to_replay_buffer(
    replay_buffer: list[list[torch.Tensor]],
    buffer_size: int,
    state: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    done: torch.Tensor,
) -> None:
    replay_buffer.append([state, action, reward, next_state, done])
    if len(replay_buffer) > buffer_size:
        replay_buffer.pop(0)


def sample_from_replay_buffer(
    replay_buffer: list[list[torch.Tensor]],
    batch_size: int,
) -> zip[tuple[torch.Tensor, ...]]:
    return zip(*random.sample(replay_buffer, batch_size), strict=False)


########################################
#            Training loop             #
########################################


def train_policy(
    env: ChopperScape,
    num_episodes: int,
    num_noise_episodes: int,
    max_steps: int,
    gamma: float,
    tau: float,
    buffer_size: int,
    batch_size: int,
    actor_lr: float = 0.0001,
    critic_lr: float = 0.0005,
    state_size: int = 10,
    action_size: int = 2,
    critic_input_size: int = 12,
    hidden1: int = 128,
    hidden2: int = 256,
    end_factor: float = 0.05,
    noise_mu: float = 0.0,
    noise_theta: float = 0.15,
    noise_sigma: float = 0.2,
    device: torch.device | None = None,
) -> tuple[Actor, list[float]]:

    # ── Resolve the device once for every tensor below ──
    dev = device if device is not None else DEVICE

    # ── Build the online and target networks ──
    actor = Actor(state_size, action_size, hidden1, hidden2).to(dev)
    critic = Critic(critic_input_size, 1, hidden1, hidden2).to(dev)
    actor.apply(init_weights)
    critic.apply(init_weights)

    actor_target = Actor(state_size, action_size, hidden1, hidden2).to(dev)
    critic_target = Critic(critic_input_size, 1, hidden1, hidden2).to(dev)
    update_target_network(actor_target, actor, 1)
    update_target_network(critic_target, critic, 1)

    # ── Optimizers with a linearly decaying learning rate ──
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=end_factor,
        total_iters=num_episodes,
    )
    critic_scheduler = LinearLR(
        critic_optimizer,
        start_factor=1.0,
        end_factor=end_factor,
        total_iters=num_episodes,
    )

    replay_buffer: list[list[torch.Tensor]] = []

    ou_noise = OUNoise(
        action_size,
        mu=noise_mu,
        theta=noise_theta,
        sigma=noise_sigma,
        device=dev,
    )

    episode_rewards: list[float] = []

    for episode in range(num_episodes):
        state = torch.FloatTensor(env.reset()).reshape(1, -1).to(dev)
        episode_reward = 0.0
        episode_actor_loss: list[float] = []
        episode_critic_loss: list[float] = []
        noise_factor = max(0.1, 1 - episode / num_noise_episodes)

        time_step = 0
        for step_idx in range(max_steps):
            time_step = step_idx

            # ── Pick an action: exploit with noise, or explore at random ──
            if len(replay_buffer) >= batch_size:
                with torch.no_grad():
                    action = actor(state)
                noise = ou_noise.sample() * noise_factor
                action = (action + noise).clamp(-1, 1)
            else:
                action = (2 * torch.rand((1, action_size)) - 1).to(dev)

            # ── Step the environment and store the transition ──
            next_state, reward, done, arrived = env.step(
                action.detach().cpu().numpy().reshape(-1),
            )
            next_state_t = torch.FloatTensor(next_state).reshape(1, -1).to(dev)
            reward_t = torch.FloatTensor([reward]).reshape(1, 1).to(dev)

            # ── Arrival is terminal too: never bootstrap past the goal ──
            terminal = float(done or arrived)
            done_t = torch.FloatTensor([terminal]).reshape(1, 1).to(dev)
            add_to_replay_buffer(
                replay_buffer,
                buffer_size,
                state,
                action,
                reward_t,
                next_state_t,
                done_t,
            )

            if len(replay_buffer) >= batch_size:
                # ── Sample a batch and assemble the tensors ──
                batch = sample_from_replay_buffer(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = batch

                action_batch = torch.cat(actions)
                reward_batch = torch.cat(rewards)
                state_batch = torch.cat(states)
                next_state_batch = torch.cat(next_states)
                dones_batch = torch.cat(dones)
                next_target_actions = actor_target(next_state_batch)

                # ── Critic update on the smooth-L1 TD error ──
                target_q_values = reward_batch + gamma * critic_target(
                    torch.cat(
                        [next_state_batch, next_target_actions.detach()],
                        dim=1,
                    ),
                ) * (1 - dones_batch)

                q_values = critic(
                    torch.cat([state_batch, action_batch], dim=1),
                )
                critic_loss = F.smooth_l1_loss(
                    q_values,
                    target_q_values.detach(),
                )
                episode_critic_loss.append(critic_loss.item())
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()

                # ── Actor update maximising the critic's value ──
                predicted_actions = actor(state_batch)
                predicted_q_values = critic(
                    torch.cat([state_batch, predicted_actions], dim=1),
                )
                actor_loss = -torch.mean(predicted_q_values)
                episode_actor_loss.append(actor_loss.item())
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()

                # ── Soft-update the target networks ──
                update_target_network(actor_target, actor, tau)
                update_target_network(critic_target, critic, tau)

            episode_reward += float(reward_t)

            if arrived:
                print("Arrived!!!")
                break
            if done_t:
                break
            else:
                state = next_state_t

        # ── Episode bookkeeping and learning-rate decay ──
        mean_actor_loss = (
            np.mean(episode_actor_loss) if episode_actor_loss else float("nan")
        )
        mean_critic_loss = (
            np.mean(episode_critic_loss)
            if episode_critic_loss
            else float("nan")
        )
        episode_reward_value = episode_reward
        actor_lr_now = round(actor_optimizer.param_groups[0]["lr"], 6)
        critic_lr_now = round(critic_optimizer.param_groups[0]["lr"], 6)
        print(
            f"Episode № {episode}, steps = {time_step}, "
            f"episode_reward = {round(episode_reward_value, 4)}, "
            f"actor_loss = {round(mean_actor_loss, 4)}, "
            f"critic_loss = {round(mean_critic_loss, 4)} "
            f"actor_lr = {actor_lr_now}, "
            f"critic_lr = {critic_lr_now}",
        )
        actor_scheduler.step()
        critic_scheduler.step()

        episode_rewards.append(episode_reward_value)

    return actor, episode_rewards


########################################
#             Entry point              #
########################################


if __name__ == "__main__":
    out_dir = "trained_models"

    env = ChopperScape()

    policy, rewards = train_policy(
        env,
        num_episodes=500,
        num_noise_episodes=400,
        gamma=0.99,
        tau=0.005,
        max_steps=500,
        buffer_size=50000,
        batch_size=256,
    )

    os.makedirs(out_dir, exist_ok=True)
    checkpoint = {"model": policy.state_dict()}
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "my-DDPG-ckpt.pt"))
