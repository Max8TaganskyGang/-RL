"""
GridWorld Environment с MNIST наблюдениями
Задача 3: Цвета кодируются изображениями из MNIST
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld_env import GridWorldEnv, generate_floor_colors, generate_obstacles


class GridWorldMNISTEnv(GridWorldEnv):
    """
    GridWorld Environment с наблюдениями в виде изображений MNIST.
    Каждый цвет кодируется случайным изображением соответствующей цифры из MNIST.
    """
    
    def __init__(
        self,
        grid_size: int = 5,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        obstacles: Optional[list] = None,
        num_colors: Optional[int] = None,
        floor_colors: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        reward_goal: float = 10.0,
        reward_step: float = -0.1,
        reward_obstacle: float = -1.0,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        image_size: int = 28,  # Размер изображения (28x28 или меньше, но не меньше 12x12)
        flatten: bool = False,  # Если True, возвращает flatten изображение
    ):
        """
        Инициализация среды GridWorld с MNIST наблюдениями.
        
        Args:
            image_size: Размер изображения (будет уменьшено с 28x28 если нужно)
            flatten: Если True, возвращает flatten изображение, иначе 2D массив
            Остальные параметры как в GridWorldEnv
        """
        # Инициализируем базовую среду
        super().__init__(
            grid_size=grid_size,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacles,
            num_colors=num_colors,
            floor_colors=floor_colors,
            seed=seed,
            reward_goal=reward_goal,
            reward_step=reward_step,
            reward_obstacle=reward_obstacle,
            max_steps=max_steps,
            render_mode=render_mode,
        )
        
        # Проверяем, что количество цветов не превышает 10 (0-9 для MNIST)
        if self.num_colors > 10:
            raise ValueError(f"Для MNIST кодирования максимум 10 цветов (0-9), получено: {self.num_colors}")
        
        self.image_size = max(12, min(28, image_size))  # Не меньше 12x12, не больше 28x28
        self.flatten = flatten
        
        # Загружаем MNIST датасет
        self._load_mnist()
        
        # Обновляем пространство наблюдений
        if flatten:
            obs_shape = (self.image_size * self.image_size,)
        else:
            obs_shape = (self.image_size, self.image_size)
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        
        # Словарь для хранения изображений для каждого цвета
        # color -> список изображений этого цвета
        self.color_to_images = {}
        self._prepare_color_images()
    
    def _load_mnist(self):
        """Загружает MNIST датасет."""
        try:
            # Загружаем тренировочный датасет MNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size), antialias=True),
            ])
            
            mnist_dataset = torchvision.datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transform
            )
            
            # Группируем изображения по цифрам
            self.mnist_data = {}
            for digit in range(10):
                self.mnist_data[digit] = []
            
            for image, label in mnist_dataset:
                # Конвертируем в numpy и нормализуем
                img_np = image.squeeze().numpy()  # Убираем channel dimension
                label_int = label if isinstance(label, int) else label.item()
                self.mnist_data[label_int].append(img_np)
            
            print(f"✓ MNIST загружен: {sum(len(imgs) for imgs in self.mnist_data.values())} изображений")
            
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки MNIST: {e}")
    
    def _prepare_color_images(self):
        """Подготавливает изображения для каждого цвета."""
        # Для каждого цвета (0 до num_colors-1) выбираем случайные изображения соответствующей цифры
        for color_idx in range(self.num_colors):
            digit = color_idx % 10  # Цикл по цифрам 0-9
            available_images = self.mnist_data[digit]
            
            if len(available_images) == 0:
                raise ValueError(f"Нет изображений для цифры {digit}")
            
            # Сохраняем все доступные изображения для этого цвета
            self.color_to_images[color_idx] = available_images.copy()
    
    def _get_obs(self) -> np.ndarray:
        """Возвращает текущее наблюдение (изображение MNIST для текущего цвета)."""
        color_idx = self.floor_colors[self.agent_pos[0], self.agent_pos[1]]
        
        # Выбираем случайное изображение для этого цвета
        available_images = self.color_to_images[color_idx]
        image = available_images[np.random.randint(0, len(available_images))]
        
        # Возвращаем flatten или 2D в зависимости от настройки
        if self.flatten:
            return image.flatten().astype(np.float32)
        else:
            return image.astype(np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Сброс среды."""
        obs, info = super().reset(seed=seed, options=options)
        # Переопределяем наблюдение, так как _get_obs уже возвращает MNIST изображение
        obs = self._get_obs()
        return obs, info
    
    def step(self, action: int):
        """Выполняет шаг в среде."""
        obs, reward, terminated, truncated, info = super().step(action)
        # Переопределяем наблюдение
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
    
    def render_rgb_array(self, agent_pos=None, goal_pos=None, obstacles=None):
        """Отрисовка в RGB массив для видео с MNIST изображениями."""
        
        if agent_pos is None:
            agent_pos = self.agent_pos
        if goal_pos is None:
            goal_pos = self.goal_pos
        if obstacles is None:
            obstacles = self.obstacles
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-0.5, self.grid_size + 0.5)
        ax.set_ylim(-0.5, self.grid_size + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Рисуем сетку
        for i in range(self.grid_size + 1):
            ax.axhline(i, color='black', linewidth=2)
            ax.axvline(i, color='black', linewidth=2)
        
        # Для каждой клетки показываем MNIST изображение
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = (i, j)
                
                if pos in obstacles:
                    # Препятствие - черная клетка с крестом
                    rect = patches.Rectangle(
                        (j, i), 1, 1,
                        linewidth=2, edgecolor='red',
                        facecolor='black', alpha=0.9
                    )
                    ax.add_patch(rect)
                    ax.text(j + 0.5, i + 0.5, '✕', 
                           ha='center', va='center', 
                           fontsize=20, color='red', fontweight='bold')
                else:
                    # Получаем цвет (индекс) этой клетки
                    color_idx = self.floor_colors[i, j]
                    digit = color_idx % 10
                    
                    # Получаем пример MNIST изображения для этой цифры
                    available_images = self.mnist_data[digit]
                    if len(available_images) > 0:
                        # Берем первое изображение для визуализации
                        mnist_img = available_images[0]
                        
                        # Используем imshow для отображения изображения на всю клетку
                        extent = [j, j + 1, i + 1, i]  # Правильные координаты с учетом invert_yaxis
                        ax.imshow(mnist_img, cmap='gray', extent=extent, 
                                 aspect='auto', interpolation='nearest', 
                                 alpha=0.9, vmin=0, vmax=1, origin='upper')
                        
                        # Рамка вокруг клетки
                        rect = patches.Rectangle(
                            (j, i), 1, 1,
                            linewidth=1.5, edgecolor='black',
                            facecolor='none'
                        )
                        ax.add_patch(rect)
        
        # Старт - зеленая рамка
        start_rect = patches.Rectangle(
            (agent_pos[1], agent_pos[0]), 1, 1,
            linewidth=4, edgecolor='green',
            facecolor='none', zorder=10
        )
        ax.add_patch(start_rect)
        
        # Цель - красная рамка
        goal_rect = patches.Rectangle(
            (goal_pos[1], goal_pos[0]), 1, 1,
            linewidth=4, edgecolor='red',
            facecolor='none', zorder=10
        )
        ax.add_patch(goal_rect)
        
        # Агент - синий круг
        agent_circle = patches.Circle(
            (agent_pos[1] + 0.5, agent_pos[0] + 0.5),
            0.3, color='blue', zorder=11, alpha=0.8
        )
        ax.add_patch(agent_circle)
        
        ax.set_title(f'GridWorld MNIST - Step: {self.steps}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        
        return frame

