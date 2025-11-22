"""
Скрипт для сравнения обычной и векторизованных сред
Сравнивает скорость и качество обучения
"""

import numpy as np
import torch
import time
import os
import sys
from typing import Dict, List, Tuple

# Добавляем корневую директорию в путь
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gridworld_env import GridWorldEnv
from task4.vectorized_wrapper import make_vectorized_env
from task4.vectorized_gridworld import VectorizedGridWorldEnv
from dqn import DQNAgent
from train import train_dqn
from task4.train_vectorized_simple import train_dqn_vectorized, evaluate_vectorized

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def create_env_basic(env_id: int, seed: int = 42):
    """Создает обычную (не векторизованную) среду."""
    from task2.train_task1 import create_task1_env
    env, num_colors = create_task1_env(env_id, seed=seed)
    return env, num_colors


def run_comparison(
    env_id: int = 1,
    num_envs: int = 4,
    n_episodes: int = 500,
    max_steps: int = 200,
    save_dir: str = "task4/results",
    use_wandb: bool = False,
    wandb_project: str = "RL4",
):
    """
    Запускает сравнение обычной, наивной и полноценной векторизованных сред.
    
    Args:
        env_id: ID окружения (1, 2, или 3)
        num_envs: Количество параллельных сред
        n_episodes: Количество эпизодов для обучения (суммарно)
        max_steps: Максимальное количество шагов в эпизоде
        save_dir: Директория для сохранения результатов
        use_wandb: Использовать ли Wandb
        wandb_project: Название проекта в Wandb
    """
    print("="*60)
    print("СРАВНЕНИЕ ОБЫЧНОЙ И ВЕКТОРИЗОВАННЫХ СРЕД")
    print("="*60)
    print(f"Окружение: {env_id}")
    print(f"Количество параллельных сред: {num_envs}")
    print(f"Количество эпизодов: {n_episodes}")
    print()
    
    results = {}
    
    # Создаем базовое окружение для получения параметров
    base_env, num_colors = create_env_basic(env_id, seed=42)
    
    # Получаем параметры окружения
    grid_size = base_env.grid_size
    start_pos = base_env.start_pos
    goal_pos = base_env.goal_pos
    obstacles = base_env.obstacles
    floor_colors = base_env.floor_colors
    
    print(f"📊 Параметры окружения:")
    print(f"   Размер сетки: {grid_size}x{grid_size}")
    print(f"   Количество цветов: {num_colors}")
    print(f"   Количество препятствий: {len(obstacles)}")
    print()
    
    # 1. Обычная среда (не векторизованная)
    print("="*60)
    print("1. ОБЫЧНАЯ СРЕДА (не векторизованная)")
    print("="*60)
    
    env_normal = GridWorldEnv(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
        floor_colors=floor_colors,
        seed=42,
        max_steps=max_steps,
    )
    
    agent_normal = DQNAgent(
        obs_dim=1,
        action_dim=4,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    start_time = time.time()
    
    metrics_normal = train_dqn(
        env_normal,
        agent_normal,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps,
        train_freq=4,
        eval_freq=50,
        save_dir=None,
        use_wandb=use_wandb,
        wandb_config={
            'project': wandb_project,
            'name': f'normal-env{env_id}',
            'config': {
                'env_id': env_id,
                'num_envs': 1,
                'vectorized': False,
            }
        } if use_wandb else None,
    )
    
    time_normal = time.time() - start_time
    # Для обычной среды total_steps = количество эпизодов * средняя длина эпизода
    total_steps_normal = sum(metrics_normal['episode_lengths']) if metrics_normal['episode_lengths'] else 0
    
    results['normal'] = {
        'metrics': metrics_normal,
        'time': time_normal,
        'total_steps': total_steps_normal,
        'steps_per_second': total_steps_normal / time_normal if time_normal > 0 else 0,
        'final_reward': np.mean(metrics_normal['episode_rewards'][-100:]) if len(metrics_normal['episode_rewards']) >= 100 else 0,
        'num_episodes': len(metrics_normal['episode_rewards']),
    }
    
    print(f"✅ Обычная среда завершена")
    print(f"   Время: {time_normal:.2f}s")
    print(f"   Всего шагов: {total_steps_normal:.0f}")
    print(f"   Шагов в секунду: {total_steps_normal / time_normal:.2f}")
    print()
    
    # 2. Наивная векторизация (SyncVectorEnv)
    print("="*60)
    print("2. НАИВНАЯ ВЕКТОРИЗАЦИЯ (SyncVectorEnv)")
    print("="*60)
    
    env_sync = make_vectorized_env(
        num_envs=num_envs,
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
        floor_colors=floor_colors,
        seed=42,
        max_steps=max_steps,
    )
    
    agent_sync = DQNAgent(
        obs_dim=1,
        action_dim=4,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    start_time_sync = time.time()
    
    metrics_sync = train_dqn_vectorized(
        env_sync,
        agent_sync,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps,
        train_freq=4,
        eval_freq=50,
        save_dir=None,
        use_wandb=use_wandb,
        wandb_config={
            'project': wandb_project,
            'name': f'sync-vectorized-env{env_id}',
            'config': {
                'env_id': env_id,
                'num_envs': num_envs,
                'vectorized': True,
                'vectorization_type': 'sync',
            }
        } if use_wandb else None,
    )
    
    time_sync = time.time() - start_time_sync
    # Для векторизованной среды total_steps = сумма по всем эпизодам и всем средам
    # Каждый эпизод дает num_envs шагов (потому что параллельно)
    if 'total_steps' in metrics_sync and metrics_sync['total_steps']:
        # Используем последнее значение total_steps (которое содержит накопленную сумму)
        total_steps_sync = metrics_sync['total_steps'][-1] if metrics_sync['total_steps'] else 0
    else:
        # Фоллбэк: сумма длин эпизодов * num_envs
        total_steps_sync = sum(metrics_sync['episode_lengths']) * num_envs if metrics_sync['episode_lengths'] else 0
    
    results['sync'] = {
        'metrics': metrics_sync,
        'time': time_sync,
        'total_steps': total_steps_sync,
        'steps_per_second': total_steps_sync / time_sync if time_sync > 0 else 0,
        'final_reward': np.mean(metrics_sync['episode_rewards'][-100:]) if len(metrics_sync['episode_rewards']) >= 100 else 0,
        'num_episodes': len(metrics_sync['episode_rewards']),
        'speedup': time_normal / time_sync if time_sync > 0 else 0,
    }
    
    print(f"✅ SyncVectorEnv завершена")
    print(f"   Время: {time_sync:.2f}s")
    print(f"   Всего шагов: {total_steps_sync:.0f}")
    print(f"   Шагов в секунду: {total_steps_sync / time_sync:.2f}")
    print(f"   Ускорение: {time_normal / time_sync:.2f}x")
    print()
    
    # 3. Полноценная векторизация (numpy)
    print("="*60)
    print("3. ПОЛНОЦЕННАЯ ВЕКТОРИЗАЦИЯ (numpy)")
    print("="*60)
    
    env_vectorized = VectorizedGridWorldEnv(
        num_envs=num_envs,
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
        floor_colors=floor_colors,
        seed=42,
        max_steps=max_steps,
    )
    
    agent_vectorized = DQNAgent(
        obs_dim=1,
        action_dim=4,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    start_time_vectorized = time.time()
    
    metrics_vectorized = train_dqn_vectorized(
        env_vectorized,
        agent_vectorized,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps,
        train_freq=4,
        eval_freq=50,
        save_dir=None,
        use_wandb=use_wandb,
        wandb_config={
            'project': wandb_project,
            'name': f'full-vectorized-env{env_id}',
            'config': {
                'env_id': env_id,
                'num_envs': num_envs,
                'vectorized': True,
                'vectorization_type': 'full',
            }
        } if use_wandb else None,
    )
    
    time_vectorized = time.time() - start_time_vectorized
    # Для полноценной векторизации также используем total_steps
    if 'total_steps' in metrics_vectorized and metrics_vectorized['total_steps']:
        total_steps_vectorized = metrics_vectorized['total_steps'][-1] if metrics_vectorized['total_steps'] else 0
    else:
        total_steps_vectorized = sum(metrics_vectorized['episode_lengths']) * num_envs if metrics_vectorized['episode_lengths'] else 0
    
    results['full'] = {
        'metrics': metrics_vectorized,
        'time': time_vectorized,
        'total_steps': total_steps_vectorized,
        'steps_per_second': total_steps_vectorized / time_vectorized if time_vectorized > 0 else 0,
        'final_reward': np.mean(metrics_vectorized['episode_rewards'][-100:]) if len(metrics_vectorized['episode_rewards']) >= 100 else 0,
        'num_episodes': len(metrics_vectorized['episode_rewards']),
        'speedup': time_normal / time_vectorized if time_vectorized > 0 else 0,
    }
    
    print(f"✅ Полноценная векторизация завершена")
    print(f"   Время: {time_vectorized:.2f}s")
    print(f"   Всего шагов: {total_steps_vectorized:.0f}")
    print(f"   Шагов в секунду: {total_steps_vectorized / time_vectorized:.2f}")
    print(f"   Ускорение: {time_normal / time_vectorized:.2f}x")
    print()
    
    # Вывод результатов сравнения
    print("="*80)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("="*80)
    print(f"{'Метод':<25} {'Время (с)':<12} {'Эпизодов':<12} {'Всего шагов':<15} {'Шагов/с':<12} {'Ускорение':<12} {'Награда':<10}")
    print("-"*100)
    
    print(f"{'Обычная':<25} {time_normal:<12.2f} {results['normal']['num_episodes']:<12} "
          f"{total_steps_normal:<15.0f} {results['normal']['steps_per_second']:<12.2f} "
          f"{'1.00x':<12} {results['normal']['final_reward']:<10.2f}")
    
    print(f"{'SyncVectorEnv':<25} {time_sync:<12.2f} {results['sync']['num_episodes']:<12} "
          f"{total_steps_sync:<15.0f} {results['sync']['steps_per_second']:<12.2f} "
          f"{results['sync']['speedup']:<12.2f}x {results['sync']['final_reward']:<10.2f}")
    
    print(f"{'Полная векторизация':<25} {time_vectorized:<12.2f} {results['full']['num_episodes']:<12} "
          f"{total_steps_vectorized:<15.0f} {results['full']['steps_per_second']:<12.2f} "
          f"{results['full']['speedup']:<12.2f}x {results['full']['final_reward']:<10.2f}")
    print()
    
    print("📊 Важные наблюдения:")
    print(f"   - SyncVectorEnv: ускорение {results['sync']['speedup']:.2f}x, "
          f"всего шагов: {total_steps_sync:.0f} (против {total_steps_normal:.0f} для обычной)")
    print(f"   - Полная векторизация: ускорение {results['full']['speedup']:.2f}x, "
          f"всего шагов: {total_steps_vectorized:.0f}")
    print()
    
    # Сохранение результатов
    os.makedirs(save_dir, exist_ok=True)
    import pickle
    with open(os.path.join(save_dir, f'comparison_env{env_id}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Сравнение обычной и векторизованных сред')
    parser.add_argument('--env', type=int, default=1, help='ID окружения (1, 2, или 3)')
    parser.add_argument('--num-envs', type=int, default=4, help='Количество параллельных сред')
    parser.add_argument('--episodes', type=int, default=500, help='Количество эпизодов')
    parser.add_argument('--wandb', action='store_true', help='Использовать Wandb')
    parser.add_argument('--wandb-project', type=str, default='RL4', help='Название проекта в Wandb')
    
    args = parser.parse_args()
    
    run_comparison(
        env_id=args.env,
        num_envs=args.num_envs,
        n_episodes=args.episodes,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

