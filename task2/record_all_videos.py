"""
Скрипт для записи видео всех обученных агентов
"""

import os
import sys
import torch
import numpy as np

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld_env import GridWorldEnv, generate_floor_colors, generate_obstacles
from dqn import DQNAgent
from ppo import PPOAgent
from visualize import record_episode_video


def create_task1_env(env_id: int, seed: int = 42):
    """Создает одно из 4 окружений из задания 1."""
    if env_id == 1:
        grid_size = 5
        start_pos = (0, 0)
        goal_pos = (4, 4)
        obstacles = []
        floor_colors = generate_floor_colors(grid_size, num_colors=None, seed=seed)
        num_colors = 25
    elif env_id == 2:
        grid_size = 5
        start_pos = (0, 0)
        goal_pos = (4, 4)
        obstacles = []
        floor_colors = generate_floor_colors(grid_size, num_colors=5, seed=seed)
        num_colors = 5
    elif env_id == 3:
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
    
    return env


def record_video(method, use_lstm, env_id, model_dir, output_dir):
    """Записать видео для одного агента."""
    env = create_task1_env(env_id, seed=42)
    obs_dim = 1
    action_dim = 4
    
    # Определяем путь к модели
    lstm_suffix = "lstm" if use_lstm else "no-lstm"
    model_path = os.path.join(model_dir, f"{method}-{lstm_suffix}", f"env{env_id}")
    
    # Ищем последнюю сохраненную модель
    if not os.path.exists(model_path):
        print(f"⚠️  Модель не найдена: {model_path}")
        env.close()
        return False
    
    # Ищем файлы моделей
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    if not model_files:
        print(f"⚠️  Файлы моделей не найдены в: {model_path}")
        env.close()
        return False
    
    # Берем последнюю модель (с наибольшим номером эпизода)
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    latest_model = os.path.join(model_path, model_files[-1])
    
    print(f"Загрузка модели: {latest_model}")
    
    # Создаем агента
    if method == 'dqn':
        agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            use_lstm=use_lstm,
            lstm_hidden_dim=128 if use_lstm else None,
            num_layers=1 if use_lstm else None
        )
    else:  # ppo
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            use_lstm=use_lstm,
            lstm_hidden_dim=128 if use_lstm else None,
            num_layers=1 if use_lstm else None
        )
    
    try:
        agent.load(latest_model)
        print(f"✓ Модель загружена")
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        env.close()
        return False
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Имя файла видео
    video_name = f"{method}-{lstm_suffix}-env{env_id}.mp4"
    video_path = os.path.join(output_dir, video_name)
    
    print(f"Запись видео: {video_path}")
    
    try:
        record_episode_video(env, agent, method=method, save_path=video_path, fps=2)
        print(f"✓ Видео сохранено: {video_path}")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Ошибка записи видео: {e}")
        env.close()
        return False


def main():
    """Записать видео для всех агентов."""
    model_dir = "task2/results"
    output_dir = "task2/task2_videos"
    
    experiments = [
        # DQN без LSTM
        ("dqn", False, 1),
        ("dqn", False, 2),
        ("dqn", False, 3),
        # DQN с LSTM
        ("dqn", True, 1),
        ("dqn", True, 2),
        ("dqn", True, 3),
        # PPO без LSTM
        ("ppo", False, 1),
        ("ppo", False, 2),
        ("ppo", False, 3),
        # PPO с LSTM
        ("ppo", True, 1),
        ("ppo", True, 2),
        ("ppo", True, 3),
    ]
    
    print("="*60)
    print("Запись видео для всех обученных агентов")
    print("="*60)
    
    results = []
    for method, use_lstm, env_id in experiments:
        success = record_video(method, use_lstm, env_id, model_dir, output_dir)
        results.append((method, use_lstm, env_id, success))
        print()
    
    print("="*60)
    print("Результаты:")
    print("="*60)
    for method, use_lstm, env_id, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {method} {'LSTM' if use_lstm else 'no-LSTM'} env{env_id}")
    
    print(f"\nВидео сохранены в: {output_dir}/")


if __name__ == "__main__":
    main()

