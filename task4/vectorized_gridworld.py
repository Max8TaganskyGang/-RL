"""
Векторизованная среда GridWorld
Полноценная векторизация с использованием numpy для параллельного выполнения
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any, List
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gridworld_env import generate_floor_colors, generate_obstacles


class VectorizedGridWorldEnv(gym.Env):
    """
    Полноценная векторизованная среда GridWorld.
    Все вычисления выполняются в batch-режиме с использованием numpy.
    
    Это НАСТОЯЩАЯ векторизация, а не просто обертка над несколькими средами.
    Все состояния, действия и переходы обрабатываются параллельно.
    """
    
    metadata = {"render_modes": [], "render_fps": 4}
    
    def __init__(
        self,
        num_envs: int = 4,
        grid_size: int = 5,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        num_colors: Optional[int] = None,
        floor_colors: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        reward_goal: float = 10.0,
        reward_step: float = -0.1,
        reward_obstacle: float = -1.0,
        max_steps: int = 100,
    ):
        """
        Инициализация векторизованной среды.
        
        Args:
            num_envs: Количество параллельных сред
            grid_size: Размер сетки
            start_pos: Начальная позиция (одинаковая для всех сред)
            goal_pos: Позиция цели (одинаковая для всех сред)
            obstacles: Список препятствий (одинаковые для всех сред)
            num_colors: Количество цветов
            floor_colors: Предзаданная карта цветов
            seed: Seed для воспроизводимости
            reward_goal: Награда за достижение цели
            reward_step: Награда за шаг
            reward_obstacle: Награда за препятствие
            max_steps: Максимальное количество шагов
        """
        super().__init__()
        
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_obstacle = reward_obstacle
        self.max_steps = max_steps
        
        # Препятствия и позиции (одинаковые для всех сред)
        self.obstacles = set(obstacles) if obstacles else set()
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        
        # Генерируем карту цветов
        if floor_colors is not None:
            self.floor_colors = floor_colors
            self.num_colors = int(np.max(floor_colors) + 1)
        else:
            self.floor_colors = generate_floor_colors(grid_size, num_colors, seed=seed)
            if num_colors is None:
                self.num_colors = grid_size * grid_size
            else:
                self.num_colors = num_colors
        
        # Пространство действий: 0=вверх, 1=вправо, 2=вниз, 3=влево
        self.action_space = spaces.Discrete(4)
        
        # Пространство наблюдений: индекс цвета (скаляр)
        self.observation_space = spaces.Box(
            low=0, high=self.num_colors - 1, shape=(1,), dtype=np.int32
        )
        
        # Матрица препятствий для быстрой проверки
        self.obstacle_mask = np.zeros((grid_size, grid_size), dtype=bool)
        for obs_pos in self.obstacles:
            self.obstacle_mask[obs_pos[0], obs_pos[1]] = True
        
        # Внутреннее состояние для всех сред (batch)
        self.agent_positions = None  # Shape: (num_envs, 2) - [row, col]
        self.steps = None  # Shape: (num_envs,) - количество шагов для каждой среды
        self.done = None  # Shape: (num_envs,) - флаг завершения для каждой среды
        
        # Движения для всех направлений
        self.moves = np.array([
            [-1, 0],  # вверх (0)
            [0, 1],   # вправо (1)
            [1, 0],   # вниз (2)
            [0, -1],  # влево (3)
        ], dtype=np.int32)
        
        # Seed для воспроизводимости
        self.np_random = np.random.RandomState(seed)
    
    def _get_obs(self) -> np.ndarray:
        """Возвращает наблюдения для всех сред."""
        # Shape: (num_envs, 1)
        rows = self.agent_positions[:, 0]
        cols = self.agent_positions[:, 1]
        color_indices = self.floor_colors[rows, cols]
        return color_indices.reshape(self.num_envs, 1).astype(np.int32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Возвращает информацию о состоянии всех сред."""
        goal_pos_arr = np.array(self.goal_pos) if self.goal_pos else None
        
        if goal_pos_arr is not None:
            # Вычисляем расстояния до цели для всех сред
            distances = np.abs(self.agent_positions[:, 0] - goal_pos_arr[0]) + \
                       np.abs(self.agent_positions[:, 1] - goal_pos_arr[1])
        else:
            distances = None
        
        return {
            "agent_positions": self.agent_positions.copy(),
            "steps": self.steps.copy(),
            "distances_to_goal": distances,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Сброс всех сред в начальное состояние."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Инициализируем массивы состояний
        self.agent_positions = np.zeros((self.num_envs, 2), dtype=np.int32)
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.done = np.zeros(self.num_envs, dtype=bool)
        
        # Устанавливаем цель, если не задана (делаем это один раз)
        if self.goal_pos is None:
            if not hasattr(self, '_cached_goal_pos'):
                # Выбираем случайную цель (одинаковую для всех сред)
                while True:
                    goal_pos = tuple(self.np_random.randint(0, self.grid_size, size=2))
                    if goal_pos not in self.obstacles:
                        self._cached_goal_pos = goal_pos
                        break
            self.goal_pos = self._cached_goal_pos
        
        self.goal_pos_array = np.array(self.goal_pos)
        
        # Устанавливаем начальные позиции для всех сред
        if self.start_pos is not None:
            # Все среды начинают с одной позиции
            self.agent_positions[:, :] = np.array(self.start_pos)
        else:
            # Случайные начальные позиции для каждой среды
            for i in range(self.num_envs):
                attempts = 0
                while attempts < 100:  # Ограничиваем количество попыток
                    pos = self.np_random.randint(0, self.grid_size, size=2)
                    pos_tuple = tuple(pos)
                    if pos_tuple not in self.obstacles and tuple(pos) != tuple(self.goal_pos):
                        self.agent_positions[i] = pos
                        break
                    attempts += 1
                if attempts >= 100:
                    # Если не удалось найти позицию, используем (0, 0)
                    self.agent_positions[i] = [0, 0]
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Выполняет один шаг во всех средах параллельно.
        
        Args:
            actions: Массив действий shape (num_envs,)
        
        Returns:
            observations: (num_envs, obs_shape)
            rewards: (num_envs,)
            terminated: (num_envs,)
            truncated: (num_envs,)
            info: dict с дополнительной информацией
        """
        # Вычисляем новые позиции для всех сред одновременно
        # actions shape: (num_envs,)
        # self.moves shape: (4, 2)
        # Получаем движения для каждого действия: (num_envs, 2)
        movements = self.moves[actions]
        
        # Вычисляем новые позиции
        new_positions = self.agent_positions + movements
        
        # Ограничиваем позиции границами сетки
        new_positions = np.clip(new_positions, 0, self.grid_size - 1)
        
        # Проверяем столкновения с препятствиями (векторизованно)
        # Используем advanced indexing для эффективной проверки
        rows = new_positions[:, 0]
        cols = new_positions[:, 1]
        obstacle_hits = self.obstacle_mask[rows, cols]
        
        # Если столкнулись с препятствием, остаемся на месте
        self.agent_positions = np.where(
            obstacle_hits[:, np.newaxis],
            self.agent_positions,  # остаемся на месте
            new_positions  # двигаемся
        )
        
        # Увеличиваем счетчик шагов только для незавершенных сред
        self.steps += ~self.done
        
        # Вычисляем награды (векторизованно)
        rewards = np.full(self.num_envs, self.reward_step, dtype=np.float32)
        
        # Награда за столкновение с препятствием
        rewards[obstacle_hits] += self.reward_obstacle
        
        # Проверяем достижение цели (векторизованно)
        at_goal = np.all(self.agent_positions == self.goal_pos_array, axis=1)
        
        # Награда за достижение цели
        rewards[at_goal & ~self.done] = self.reward_goal
        
        # Обновляем флаги завершения
        terminated = at_goal & ~self.done
        self.done |= terminated
        
        # Проверяем превышение максимального количества шагов
        truncated = (self.steps >= self.max_steps) & ~self.done
        self.done |= truncated
        
        # Получаем наблюдения
        observations = self._get_obs()
        
        # Информация
        info = self._get_info()
        
        return observations, rewards, terminated, truncated, info
    
    def reset_single(self, env_idx: int, seed: Optional[int] = None):
        """Сброс одной среды по индексу."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Сбрасываем только указанную среду
        if self.start_pos is not None:
            self.agent_positions[env_idx] = np.array(self.start_pos)
        else:
            # Случайная начальная позиция
            attempts = 0
            while attempts < 100:
                pos = self.np_random.randint(0, self.grid_size, size=2)
                pos_tuple = tuple(pos)
                if pos_tuple not in self.obstacles and tuple(pos) != tuple(self.goal_pos):
                    self.agent_positions[env_idx] = pos
                    break
                attempts += 1
            if attempts >= 100:
                self.agent_positions[env_idx] = [0, 0]
        
        self.steps[env_idx] = 0
        self.done[env_idx] = False
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs[env_idx], {k: v[env_idx] if isinstance(v, np.ndarray) else v 
                             for k, v in info.items()}
    
    def close(self):
        """Закрытие среды."""
        pass

