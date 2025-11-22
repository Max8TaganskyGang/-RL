"""
Скрипт для обучения DQN и PPO на всех 4 средах из задания 1
С поддержкой LSTM и без LSTM
"""

import argparse
import os
import sys
import numpy as np

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld_env import GridWorldEnv, generate_floor_colors, generate_obstacles
from dqn import DQNAgent
from ppo import PPOAgent
from networks import QNetwork, QNetworkLSTM, ActorCritic, ActorCriticLSTM
from train import train_dqn, train_ppo

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def create_task1_env(env_id: int, seed: int = 42):
    """
    Создает одно из 4 окружений из задания 1.
    
    Args:
        env_id: ID окружения (1-4)
        seed: Seed для воспроизводимости
    
    Returns:
        GridWorldEnv: Созданное окружение
    """
    if env_id == 1:
        # Окружение 1: 5x5 с уникальными цветами для каждой позиции (25 цветов) без препятствий
        grid_size = 5
        start_pos = (0, 0)
        goal_pos = (4, 4)
        obstacles = []
        floor_colors = generate_floor_colors(grid_size, num_colors=None, seed=seed)
        num_colors = 25
    elif env_id == 2:
        # Окружение 2: 5x5 с 5 цветами пола без препятствий
        grid_size = 5
        start_pos = (0, 0)
        goal_pos = (4, 4)
        obstacles = []
        floor_colors = generate_floor_colors(grid_size, num_colors=5, seed=seed)
        num_colors = 5
    elif env_id == 3:
        # Окружение 3: 10x10 с 7 цветами пола и 10% препятствий
        grid_size = 10
        start_pos = (0, 0)
        goal_pos = (9, 9)
        obstacles = generate_obstacles(
            grid_size, 
            obstacle_percentage=0.1, 
            seed=seed,
            exclude_positions=[start_pos, goal_pos]
        )
        floor_colors = generate_floor_colors(grid_size, num_colors=7, seed=seed)
        num_colors = 7
    elif env_id == 4:
        # Окружение 4: 10x10 с 4 цветами пола и 10% препятствий
        grid_size = 10
        start_pos = (0, 0)
        goal_pos = (9, 9)
        obstacles = generate_obstacles(
            grid_size, 
            obstacle_percentage=0.1, 
            seed=seed,
            exclude_positions=[start_pos, goal_pos]
        )
        floor_colors = generate_floor_colors(grid_size, num_colors=4, seed=seed)
        num_colors = 4
    else:
        raise ValueError(f"Unknown environment ID: {env_id}")
    
    env = GridWorldEnv(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
        floor_colors=floor_colors,
        seed=seed,
        max_steps=200
    )
    
    return env, num_colors


def main():
    parser = argparse.ArgumentParser(description='Train DQN or PPO on Task 1 environments')
    parser.add_argument('--method', type=str, choices=['dqn', 'ppo', 'both'], 
                       default='both', help='Method to train')
    parser.add_argument('--env', type=int, choices=[1, 2, 3, 4], 
                       default=None, help='Environment ID (1-4). If None, train on all')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--use-lstm', action='store_true', help='Use LSTM networks')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='RL2', help='Wandb project name')
    
    args = parser.parse_args()
    
    env_ids = [args.env] if args.env else [1, 2, 3, 4]
    methods = ['dqn', 'ppo'] if args.method == 'both' else [args.method]
    
    for env_id in env_ids:
        print(f"\n{'='*60}")
        print(f"Окружение {env_id}")
        print(f"{'='*60}")
        
        env, num_colors = create_task1_env(env_id, seed=args.seed)
        shortest_path = env.get_shortest_path_length()
        print(f"Кратчайший путь: {shortest_path} шагов")
        
        obs_dim = 1  # Индекс цвета
        action_dim = 4
        
        for method in methods:
            print(f"\n--- {method.upper()}{' с LSTM' if args.use_lstm else ''} ---")
            
            if method == 'dqn':
                if args.use_lstm:
                    # DQN с LSTM
                    agent = DQNAgent(
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        lr=1e-3,
                        gamma=0.99,
                        epsilon_start=1.0,
                        epsilon_end=0.01,
                        epsilon_decay=0.995,
                        buffer_size=10000,
                        batch_size=64,
                        target_update_freq=100,
                        use_lstm=True,
                        lstm_hidden_dim=128,
                        num_layers=1
                    )
                else:
                    # DQN без LSTM
                    agent = DQNAgent(
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        lr=1e-3,
                        gamma=0.99,
                        epsilon_start=1.0,
                        epsilon_end=0.01,
                        epsilon_decay=0.995,
                        buffer_size=10000,
                        batch_size=64,
                        target_update_freq=100,
                        use_lstm=False
                    )
                
                wandb_config = None
                if args.wandb and WANDB_AVAILABLE:
                    wandb_config = {
                        'project': args.wandb_project,
                        'name': f'dqn-{"lstm" if args.use_lstm else "no-lstm"}-env{env_id}',
                        'config': {
                            'env_id': env_id,
                            'method': 'dqn',
                            'use_lstm': args.use_lstm,
                            'n_episodes': args.episodes,
                            'seed': args.seed,
                            'num_colors': num_colors,
                            'shortest_path': shortest_path,
                            'obs_dim': obs_dim,
                            'action_dim': action_dim,
                        }
                    }
                
                save_dir = f"task2/results/dqn-{'lstm' if args.use_lstm else 'no-lstm'}/env{env_id}"
                os.makedirs(save_dir, exist_ok=True)
                
                train_dqn(
                    env=env,
                    agent=agent,
                    n_episodes=args.episodes,
                    max_steps_per_episode=200,
                    train_freq=4,
                    eval_freq=50,
                    eval_episodes=10,
                    save_dir=save_dir,
                    use_wandb=args.wandb,
                    wandb_config=wandb_config
                )
                
                # Закрываем wandb сессию
                if args.wandb and WANDB_AVAILABLE:
                    wandb.finish()
                
            elif method == 'ppo':
                if args.use_lstm:
                    # PPO с LSTM
                    agent = PPOAgent(
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        lr=3e-4,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_epsilon=0.2,
                        value_coef=0.5,
                        entropy_coef=0.01,
                        max_grad_norm=0.5,
                        use_lstm=True,
                        lstm_hidden_dim=128,
                        num_layers=1
                    )
                else:
                    # PPO без LSTM
                    agent = PPOAgent(
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        lr=3e-4,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_epsilon=0.2,
                        value_coef=0.5,
                        entropy_coef=0.01,
                        max_grad_norm=0.5,
                        use_lstm=False
                    )
                
                wandb_config = None
                if args.wandb and WANDB_AVAILABLE:
                    wandb_config = {
                        'project': args.wandb_project,
                        'name': f'ppo-{"lstm" if args.use_lstm else "no-lstm"}-env{env_id}',
                        'config': {
                            'env_id': env_id,
                            'method': 'ppo',
                            'use_lstm': args.use_lstm,
                            'n_episodes': args.episodes,
                            'seed': args.seed,
                            'num_colors': num_colors,
                            'shortest_path': shortest_path,
                            'obs_dim': obs_dim,
                            'action_dim': action_dim,
                        }
                    }
                
                save_dir = f"task2/results/ppo-{'lstm' if args.use_lstm else 'no-lstm'}/env{env_id}"
                os.makedirs(save_dir, exist_ok=True)
                
                train_ppo(
                    env=env,
                    agent=agent,
                    n_episodes=args.episodes,
                    max_steps_per_episode=200,
                    update_freq=200,  # Обновление каждые ~200 шагов
                    n_epochs=10,
                    batch_size=64,
                    eval_freq=50,
                    eval_episodes=10,
                    save_dir=save_dir,
                    use_wandb=args.wandb,
                    wandb_config=wandb_config
                )
                
                # Закрываем wandb сессию
                if args.wandb and WANDB_AVAILABLE:
                    wandb.finish()
        
        env.close()
    
    print("\n" + "="*60)
    print("Обучение завершено!")
    print("="*60)


if __name__ == "__main__":
    main()

