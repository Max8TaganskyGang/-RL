"""
Скрипт для проведения экспериментов с DQN и PPO
"""

import os
import argparse
from gridworld_env import GridWorldEnv
from dqn import DQNAgent
from ppo import PPOAgent
from train import train_dqn, train_ppo
from visualize import plot_training_curves, visualize_episode, record_episode_video
from compare_methods import compare_training_curves, print_comparison_summary


def create_env(env_name: str):
    """Создать окружение по имени."""
    if env_name == 'simple':
        return GridWorldEnv(
            grid_size=5,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            obstacles=[(1, 1), (2, 2), (3, 3)],
            render_mode=None
        )
    elif env_name == 'easy':
        return GridWorldEnv(
            grid_size=5,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            obstacles=[],
            render_mode=None
        )
    elif env_name == 'medium':
        return GridWorldEnv(
            grid_size=5,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            obstacles=[(1, 2), (2, 2), (3, 2)],
            render_mode=None
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def run_dqn_experiment(env_name: str, n_episodes: int = 500, save_dir: str = "results/dqn", 
                       use_wandb: bool = False, wandb_project: str = "gridworld-dqn"):
    """Запустить эксперимент с DQN."""
    print("=" * 60)
    print(f"DQN Experiment: {env_name}")
    print("=" * 60)
    
    env = create_env(env_name)
    agent = DQNAgent(
        obs_dim=2,
        action_dim=4,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
    )
    
    wandb_config = None
    if use_wandb:
        wandb_config = {
            'project': wandb_project,
            'name': f'dqn-{env_name}',
            'config': {
                'method': 'DQN',
                'env': env_name,
                'n_episodes': n_episodes,
                'lr': 1e-3,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995,
                'buffer_size': 10000,
                'batch_size': 64,
                'target_update_freq': 100,
            }
        }
    
    metrics = train_dqn(
        env=env,
        agent=agent,
        n_episodes=n_episodes,
        max_steps_per_episode=200,
        train_freq=4,
        eval_freq=50,
        eval_episodes=10,
        save_dir=save_dir,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
    )
    
    # Визуализация
    os.makedirs(save_dir, exist_ok=True)
    plot_training_curves(
        metrics,
        save_path=os.path.join(save_dir, f'training_curves_{env_name}.png'),
        title=f'DQN Training - {env_name}'
    )
    
    # Визуализация эпизода
    visualize_episode(
        env,
        agent,
        method='dqn',
        save_path=os.path.join(save_dir, f'trajectory_{env_name}.png')
    )
    
    # Запись видео эпизода
    record_episode_video(
        env,
        agent,
        method='dqn',
        save_path=os.path.join(save_dir, f'episode_{env_name}.mp4'),
        fps=2
    )
    
    env.close()
    return metrics


def run_ppo_experiment(env_name: str, n_episodes: int = 500, save_dir: str = "results/ppo",
                       use_wandb: bool = False, wandb_project: str = "gridworld-ppo"):
    """Запустить эксперимент с PPO."""
    print("=" * 60)
    print(f"PPO Experiment: {env_name}")
    print("=" * 60)
    
    env = create_env(env_name)
    agent = PPOAgent(
        obs_dim=2,
        action_dim=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )
    
    wandb_config = None
    if use_wandb:
        wandb_config = {
            'project': wandb_project,
            'name': f'ppo-{env_name}',
            'config': {
                'method': 'PPO',
                'env': env_name,
                'n_episodes': n_episodes,
                'lr': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'value_coef': 0.5,
                'entropy_coef': 0.01,
                'update_freq': 512,
                'n_epochs': 10,
                'batch_size': 64,
            }
        }
    
    metrics = train_ppo(
        env=env,
        agent=agent,
        n_episodes=n_episodes,
        max_steps_per_episode=200,
        update_freq=512,
        n_epochs=10,
        batch_size=64,
        eval_freq=50,
        eval_episodes=10,
        save_dir=save_dir,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
    )
    
    # Визуализация
    os.makedirs(save_dir, exist_ok=True)
    plot_training_curves(
        metrics,
        save_path=os.path.join(save_dir, f'training_curves_{env_name}.png'),
        title=f'PPO Training - {env_name}'
    )
    
    # Визуализация эпизода
    visualize_episode(
        env,
        agent,
        method='ppo',
        save_path=os.path.join(save_dir, f'trajectory_{env_name}.png')
    )
    
    # Запись видео эпизода
    record_episode_video(
        env,
        agent,
        method='ppo',
        save_path=os.path.join(save_dir, f'episode_{env_name}.mp4'),
        fps=2
    )
    
    env.close()
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train DQN or PPO on GridWorld')
    parser.add_argument('--method', type=str, choices=['dqn', 'ppo', 'both'], 
                       default='both', help='Method to use')
    parser.add_argument('--env', type=str, choices=['simple', 'easy', 'medium'],
                       default='simple', help='Environment to use')
    parser.add_argument('--episodes', type=int, default=500, 
                       help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb', action='store_true', 
                       help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='gridworld-rl',
                       help='Wandb project name')
    
    args = parser.parse_args()
    
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    dqn_metrics = None
    ppo_metrics = None
    
    if args.method in ['dqn', 'both']:
        dqn_metrics = run_dqn_experiment(
            args.env, 
            args.episodes, 
            use_wandb=args.wandb,
            wandb_project=args.wandb_project
        )
    
    if args.method in ['ppo', 'both']:
        ppo_metrics = run_ppo_experiment(
            args.env, 
            args.episodes,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project
        )
    
    # Сравнительный анализ
    if args.method == 'both' and dqn_metrics and ppo_metrics:
        compare_training_curves(
            dqn_metrics,
            ppo_metrics,
            save_path=f"results/comparison_{args.env}.png"
        )
        print_comparison_summary(dqn_metrics, ppo_metrics)


if __name__ == "__main__":
    main()

