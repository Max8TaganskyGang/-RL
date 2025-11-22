"""
Скрипт для обучения DQN и PPO на средах с MNIST наблюдениями (Задача 3)
"""

import argparse
import os
import sys
import numpy as np

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task3.gridworld_mnist_env import GridWorldMNISTEnv
from gridworld_env import generate_floor_colors, generate_obstacles
from dqn import DQNAgent
from ppo import PPOAgent
from train import train_dqn, train_ppo

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def create_task3_env(env_id: int, seed: int = 42, image_size: int = 28, flatten: bool = False):
    """
    Создает окружения для задачи 3 с MNIST наблюдениями.
    Максимум 7 цветов пола (+3 служебных) = максимум 10 цветов.
    
    Args:
        env_id: ID окружения (2, 3, 4) - env1 нельзя использовать (25 цветов > 10)
        seed: Seed для воспроизводимости
        image_size: Размер изображения MNIST (12-28)
        flatten: Если True, возвращает flatten изображение
    """
    if env_id == 1:
        # Окружение 1: 3x3 с уникальным числом на каждую клетку (9 цветов)
        grid_size = 3
        start_pos = (0, 0)
        goal_pos = (2, 2)
        obstacles = []
        # Создаем уникальные цвета для каждой клетки (0-8)
        floor_colors = np.zeros((grid_size, grid_size), dtype=int)
        color_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) != goal_pos:
                    color_idx += 1
                    floor_colors[i, j] = color_idx
        # Цвет 0 только у цели
        floor_colors[goal_pos[0], goal_pos[1]] = 0
        num_colors = 9  # 0-8
    elif env_id == 2:
        # Окружение 2: 5x5 с 5 цветами пола без препятствий
        grid_size = 5
        start_pos = (0, 0)
        goal_pos = (4, 4)
        obstacles = []
        floor_colors = generate_floor_colors(grid_size, num_colors=5, seed=seed)
        # Цвет 0 только у цели
        floor_colors[goal_pos[0], goal_pos[1]] = 0
        # Остальные цвета от 1 до 4
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) != goal_pos:
                    if floor_colors[i, j] == 0:
                        floor_colors[i, j] = 1
        num_colors = 5
    elif env_id == 2:
        # Окружение 2: 10x10 с 7 цветами пола и 10% препятствий
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
        # Цвет 0 только у цели
        floor_colors[goal_pos[0], goal_pos[1]] = 0
        # Остальные цвета от 1 до 6
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) != goal_pos and (i, j) not in obstacles:
                    if floor_colors[i, j] == 0:
                        floor_colors[i, j] = 1
        num_colors = 7
    elif env_id == 3:
        # Окружение 3: 10x10 с 4 цветами пола и 10% препятствий
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
        # Цвет 0 только у цели
        floor_colors[goal_pos[0], goal_pos[1]] = 0
        # Остальные цвета от 1 до 3
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) != goal_pos and (i, j) not in obstacles:
                    if floor_colors[i, j] == 0:
                        floor_colors[i, j] = 1
        num_colors = 4
    else:
        raise ValueError(f"Unknown environment ID: {env_id}")
    
    env = GridWorldMNISTEnv(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
        floor_colors=floor_colors,
        seed=seed,
        max_steps=50,
        image_size=image_size,
        flatten=flatten
    )
    
    return env, num_colors


def main():
    parser = argparse.ArgumentParser(description='Train DQN or PPO on Task 3 (MNIST) environments')
    parser.add_argument('--method', type=str, choices=['dqn', 'ppo', 'both'], 
                       default='both', help='Method to train')
    parser.add_argument('--env', type=int, choices=[1, 2, 3], 
                       default=None, help='Environment ID (1-3). If None, train on all')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--image-size', type=int, default=28, help='MNIST image size (12-28)')
    parser.add_argument('--flatten', action='store_true', help='Use flatten observations')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='RL3', help='Wandb project name')
    
    args = parser.parse_args()
    
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Определяем окружения для обучения
    env_ids_to_run = [1, 2, 3] if args.env is None else [args.env]
    methods_to_run = ['dqn', 'ppo'] if args.method == 'both' else [args.method]
    
    for env_id in env_ids_to_run:
        print(f"\n{'='*60}")
        print(f"Окружение {env_id}")
        print(f"{'='*60}")
        
        env, num_colors = create_task3_env(env_id, args.seed, args.image_size, args.flatten)
        
        # Определяем размерность наблюдения
        if args.flatten:
            obs_dim = args.image_size * args.image_size
        else:
            obs_dim = (args.image_size, args.image_size)
        
        action_dim = env.action_space.n
        
        print(f"Размерность наблюдения: {obs_dim}")
        print(f"Количество действий: {action_dim}")
        print(f"Количество цветов: {num_colors}")
        
        for method in methods_to_run:
            print(f"\n{'-'*60}")
            print(f"Метод: {method.upper()}")
            print(f"{'-'*60}")
            
            # Создаем агента
            if method == 'dqn':
                agent = DQNAgent(
                    obs_dim=obs_dim if isinstance(obs_dim, int) else obs_dim[0] * obs_dim[1],
                    action_dim=action_dim,
                    use_cnn=True,
                    image_size=args.image_size,
                )
            else:  # ppo
                agent = PPOAgent(
                    obs_dim=obs_dim if isinstance(obs_dim, int) else obs_dim[0] * obs_dim[1],
                    action_dim=action_dim,
                    use_cnn=True,
                    image_size=args.image_size,
                )
            
            # Настройка Wandb
            wandb_config = None
            if args.wandb and WANDB_AVAILABLE:
                wandb_config = {
                    'project': args.wandb_project,
                    'name': f'{method}-cnn-env{env_id}',
                    'config': {
                        'env_id': env_id,
                        'method': method,
                        'use_cnn': True,
                        'image_size': args.image_size,
                        'flatten': args.flatten,
                        'n_episodes': args.episodes,
                        'seed': args.seed,
                        'num_colors': num_colors,
                        'obs_dim': obs_dim,
                        'action_dim': action_dim,
                    }
                }
            
            # Директория для сохранения
            save_dir = f"task3/results/{method}-cnn/env{env_id}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Обучение
            if method == 'dqn':
                train_dqn(
                    env=env,
                    agent=agent,
                    n_episodes=args.episodes,
                    max_steps_per_episode=50,  # Уменьшено для быстрого обучения
                    save_dir=save_dir,
                    use_wandb=args.wandb,
                    wandb_config=wandb_config
                )
            elif method == 'ppo':
                train_ppo(
                    env=env,
                    agent=agent,
                    n_episodes=args.episodes,
                    max_steps_per_episode=50,  # Уменьшено для быстрого обучения
                    save_dir=save_dir,
                    use_wandb=args.wandb,
                    wandb_config=wandb_config,
                    update_freq=50  # Уменьшено для более частых обновлений
                )
            
            if args.wandb and WANDB_AVAILABLE:
                wandb.finish()
            
            env.close()
    
    print("\n" + "="*60)
    print("Обучение завершено!")
    print("="*60)


if __name__ == "__main__":
    main()

