"""
Скрипт для записи видео всех обученных агентов Task 3 (MNIST)
"""

import os
import sys
import torch
import numpy as np

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task3.gridworld_mnist_env import GridWorldMNISTEnv
from gridworld_env import generate_floor_colors, generate_obstacles
from dqn import DQNAgent
from ppo import PPOAgent
from visualize import record_episode_video


def create_task3_env(env_id: int, seed: int = 42, image_size: int = 28, flatten: bool = False):
    """Создает окружение для Task 3."""
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
    elif env_id == 2:
        grid_size = 5
        start_pos = (0, 0)
        goal_pos = (4, 4)
        obstacles = []
        floor_colors = generate_floor_colors(grid_size, num_colors=5, seed=seed)
        # Цвет 0 только у цели
        floor_colors[goal_pos[0], goal_pos[1]] = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) != goal_pos:
                    if floor_colors[i, j] == 0:
                        floor_colors[i, j] = 1
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
        # Цвет 0 только у цели
        floor_colors[goal_pos[0], goal_pos[1]] = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) != goal_pos and (i, j) not in obstacles:
                    if floor_colors[i, j] == 0:
                        floor_colors[i, j] = 1
    elif env_id == 4:
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
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) != goal_pos and (i, j) not in obstacles:
                    if floor_colors[i, j] == 0:
                        floor_colors[i, j] = 1
    else:
        raise ValueError(f"Unknown environment ID: {env_id}")
    
    env = GridWorldMNISTEnv(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
        floor_colors=floor_colors,
        seed=seed,
        max_steps=100,
        image_size=image_size,
        flatten=flatten
    )
    
    return env


def record_video(method, env_id, model_dir, output_dir, image_size=28, flatten=False):
    """Записать видео для одного агента."""
    env = create_task3_env(env_id, seed=42, image_size=image_size, flatten=flatten)
    
    # Определяем размерность наблюдения
    if flatten:
        obs_dim = image_size * image_size
    else:
        obs_dim = image_size * image_size  # Для CNN все равно нужен общий размер
    
    action_dim = 4
    
    # Путь к модели
    model_path = os.path.join(model_dir, "results", "task3", f"{method}-cnn", f"env{env_id}")
    
    if not os.path.exists(model_path):
        print(f"⚠️  Модель не найдена: {model_path}")
        env.close()
        return False
    
    # Ищем последнюю сохраненную модель
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    if not model_files:
        print(f"⚠️  Файлы моделей не найдены в: {model_path}")
        env.close()
        return False
    
    # Берем последнюю модель
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
    latest_model = os.path.join(model_path, model_files[-1])
    
    print(f"Загрузка модели: {latest_model}")
    
    # Создаем агента
    if method == 'dqn':
        agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            use_cnn=True,
            image_size=image_size,
        )
    else:  # ppo
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            use_cnn=True,
            image_size=image_size,
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
    video_name = f"{method}-cnn-env{env_id}.mp4"
    video_path = os.path.join(output_dir, video_name)
    
    print(f"Запись видео: {video_path}")
    
    try:
        record_episode_video(env, agent, method=method, save_path=video_path, fps=2, max_steps=100)
        print(f"✓ Видео сохранено: {video_path}")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Ошибка записи видео: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False


def main():
    """Записать видео для всех агентов Task 3."""
    model_dir = "task3"
    output_dir = "task3/task3_videos"
    
    experiments = [
        # DQN
        ("dqn", 1),
        ("dqn", 2),
        ("dqn", 3),
        # PPO
        ("ppo", 1),
        ("ppo", 2),
        ("ppo", 3),
    ]
    
    print("="*60)
    print("Запись видео для всех обученных агентов Task 3")
    print("="*60)
    
    results = []
    for method, env_id in experiments:
        print(f"\n{method.upper()} env{env_id}...")
        success = record_video(method, env_id, model_dir, output_dir, image_size=28, flatten=False)
        results.append((method, env_id, success))
        print()
    
    print("="*60)
    print("Результаты:")
    print("="*60)
    for method, env_id, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {method} env{env_id}")
    
    print(f"\nВидео сохранены в: {output_dir}/")


if __name__ == "__main__":
    main()

