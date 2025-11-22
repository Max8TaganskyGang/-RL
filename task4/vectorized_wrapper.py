"""
Обертка для создания векторизованной среды используя SyncVectorEnv (наивная векторизация)
"""

import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gridworld_env import GridWorldEnv
from gymnasium.vector import SyncVectorEnv


def make_vectorized_env(
    num_envs: int = 4,
    grid_size: int = 5,
    start_pos=None,
    goal_pos=None,
    obstacles=None,
    num_colors=None,
    floor_colors=None,
    seed: int = None,
    reward_goal: float = 10.0,
    reward_step: float = -0.1,
    reward_obstacle: float = -1.0,
    max_steps: int = 100,
) -> SyncVectorEnv:
    """
    Создает векторизованную среду используя SyncVectorEnv (наивная векторизация).
    
    Это НЕ полноценная векторизация - это просто обертка над несколькими
    независимыми средами, которые выполняются последовательно.
    """
    
    def make_env(env_idx: int):
        """Factory функция для создания одной среды."""
        def _make():
            env = GridWorldEnv(
                grid_size=grid_size,
                start_pos=start_pos,
                goal_pos=goal_pos,
                obstacles=obstacles,
                num_colors=num_colors,
                floor_colors=floor_colors,
                seed=seed + env_idx if seed is not None else None,
                reward_goal=reward_goal,
                reward_step=reward_step,
                reward_obstacle=reward_obstacle,
                max_steps=max_steps,
                render_mode=None,
            )
            return env
        return _make
    
    # Создаем список factory функций
    env_fns = [make_env(i) for i in range(num_envs)]
    
    # Создаем векторизованную среду
    env = SyncVectorEnv(env_fns)
    
    return env

