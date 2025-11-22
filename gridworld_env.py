"""
GridWorld Environment for Reinforcement Learning
Первое задание: Реализация среды GridWorld с цветами пола
Агент получает индекс цвета текущей клетки как наблюдение
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_floor_colors(grid_size, num_colors, seed=None):
    """
    Генерирует цветовую карту для пола.
    
    Args:
        grid_size: Размер сетки
        num_colors: Количество различных цветов (если None, то уникальный цвет для каждой позиции)
        seed: Seed для воспроизводимости
    
    Returns:
        2D массив с индексами цветов для каждой позиции
    """
    if seed is not None:
        np.random.seed(seed)
    
    if num_colors is None:
        # Уникальный цвет для каждой позиции
        colors = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    else:
        # Случайное распределение num_colors цветов
        colors = np.random.randint(0, num_colors, size=(grid_size, grid_size))
    
    return colors


def get_color_palette(num_colors):
    """
    Возвращает палитру цветов.
    
    Args:
        num_colors: Количество цветов
    
    Returns:
        Список RGB цветов
    """
    if num_colors <= 10:
        # Используем палитру tab10 для малого количества цветов
        cmap = plt.colormaps['tab10']
        colors = [cmap(i / max(num_colors - 1, 1))[:3] for i in range(num_colors)]
    elif num_colors <= 20:
        # Используем палитру tab20
        cmap = plt.colormaps['tab20']
        colors = [cmap(i / max(num_colors - 1, 1))[:3] for i in range(num_colors)]
    else:
        # Используем палитру hsv для большого количества цветов
        cmap = plt.colormaps['hsv']
        colors = [cmap(i / num_colors)[:3] for i in range(num_colors)]
    
    return colors


def generate_obstacles(grid_size, obstacle_percentage, seed=None, exclude_positions=None):
    """
    Генерирует случайные препятствия.
    
    Args:
        grid_size: Размер сетки
        obstacle_percentage: Процент препятствий (0-1)
        seed: Seed для воспроизводимости
        exclude_positions: Позиции, которые нужно исключить (например, старт и цель)
    
    Returns:
        Список позиций препятствий
    """
    if seed is not None:
        np.random.seed(seed)
    
    if exclude_positions is None:
        exclude_positions = []
    
    total_cells = grid_size * grid_size
    num_obstacles = int(total_cells * obstacle_percentage)
    
    # Генерируем все возможные позиции
    all_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    
    # Исключаем позиции
    available_positions = [pos for pos in all_positions if pos not in exclude_positions]
    
    # Выбираем случайные препятствия
    if num_obstacles > len(available_positions):
        num_obstacles = len(available_positions)
    
    obstacles = np.random.choice(
        len(available_positions), 
        size=num_obstacles, 
        replace=False
    )
    
    return [available_positions[i] for i in obstacles]


def render_env_with_colors(
    grid_size,
    start_pos,
    goal_pos,
    obstacles,
    floor_colors,
    filename,
    title,
    num_colors=None
):
    """
    Рендерит окружение с цветами пола в PNG файл.
    
    Args:
        grid_size: Размер сетки
        start_pos: Начальная позиция агента (row, col)
        goal_pos: Позиция цели (row, col)
        obstacles: Список позиций препятствий
        floor_colors: 2D массив с индексами цветов для каждой позиции
        filename: Имя файла для сохранения
        title: Заголовок изображения
        num_colors: Количество различных цветов (для легенды)
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Получаем палитру цветов
    if num_colors is None:
        num_colors = np.max(floor_colors) + 1
    color_palette = get_color_palette(num_colors)
    
    # Рисуем пол с цветами
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) not in obstacles:
                color_idx = floor_colors[i, j]
                color = color_palette[color_idx % len(color_palette)]
                
                rect = patches.Rectangle(
                    (j, i), 1, 1,
                    linewidth=1.5, edgecolor='black',
                    facecolor=color, alpha=0.8
                )
                ax.add_patch(rect)
                
                # Добавляем номер цвета на клетку (если это не старт и не цель)
                if (i, j) != start_pos and (i, j) != goal_pos:
                    # Определяем цвет текста в зависимости от яркости фона
                    brightness = sum(color) / 3
                    text_color = 'white' if brightness < 0.5 else 'black'
                    ax.text(j + 0.5, i + 0.5, str(color_idx), 
                           ha='center', va='center', fontsize=10, 
                           fontweight='bold', color=text_color, zorder=5)
    
    # Рисуем препятствия
    for obs in obstacles:
        rect = patches.Rectangle(
            (obs[1], obs[0]), 1, 1,
            linewidth=2, edgecolor='black',
            facecolor='gray', alpha=0.9
        )
        ax.add_patch(rect)
        # Добавляем крестик на препятствие
        ax.plot([obs[1], obs[1] + 1], [obs[0], obs[0] + 1], 
               'k-', linewidth=3, alpha=0.9, zorder=6)
        ax.plot([obs[1], obs[1] + 1], [obs[0] + 1, obs[0]], 
               'k-', linewidth=3, alpha=0.9, zorder=6)
    
    # Рисуем цель (поверх цвета пола)
    goal_rect = patches.Rectangle(
        (goal_pos[1], goal_pos[0]), 1, 1,
        linewidth=3, edgecolor='green',
        facecolor='lightgreen', alpha=0.9
    )
    ax.add_patch(goal_rect)
    
    # Рисуем агента (поверх цвета пола)
    agent_circle = patches.Circle(
        (start_pos[1] + 0.5, start_pos[0] + 0.5),
        0.35, facecolor='blue', edgecolor='darkblue', 
        zorder=10, linewidth=2
    )
    ax.add_patch(agent_circle)
    
    # Добавляем метки
    ax.text(start_pos[1] + 0.5, start_pos[0] + 0.5, 'A', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=11)
    ax.text(goal_pos[1] + 0.5, goal_pos[0] + 0.5, 'G', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='darkgreen', zorder=11)
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Чтобы (0,0) был вверху слева
    
    # Добавляем подпись о препятствиях
    full_title = f"{title}\n⚠ Поле с крестом (✕) - препятствие/опасность"
    ax.set_title(full_title, fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Column', fontsize=14)
    ax.set_ylabel('Row', fontsize=14)
    
    # Добавляем сетку
    for i in range(grid_size + 1):
        ax.axhline(i, color='black', linewidth=1.5, alpha=0.5)
        ax.axvline(i, color='black', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Сохранено: {filename}")


class GridWorldEnv(gym.Env):
    """
    Простая среда GridWorld для обучения с подкреплением.
    
    Агент находится в сетке и должен достичь целевой клетки.
    Действия: 0=вверх, 1=вправо, 2=вниз, 3=влево
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        grid_size: int = 5,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        obstacles: Optional[list] = None,
        num_colors: Optional[int] = None,  # None = уникальный цвет для каждой позиции
        floor_colors: Optional[np.ndarray] = None,  # Предзаданная карта цветов
        seed: Optional[int] = None,
        reward_goal: float = 10.0,
        reward_step: float = -0.1,
        reward_obstacle: float = -1.0,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
    ):
        """
        Инициализация среды GridWorld с цветами.
        
        Args:
            grid_size: Размер сетки (grid_size x grid_size)
            start_pos: Начальная позиция агента (row, col). Если None, выбирается случайно
            goal_pos: Позиция цели (row, col). Если None, выбирается случайно
            obstacles: Список позиций препятствий [(row, col), ...]
            num_colors: Количество цветов (None = уникальный цвет для каждой позиции)
            floor_colors: Предзаданная карта цветов (2D массив)
            seed: Seed для воспроизводимости
            reward_goal: Награда за достижение цели
            reward_step: Награда за каждый шаг (обычно отрицательная)
            reward_obstacle: Награда за столкновение с препятствием
            max_steps: Максимальное количество шагов в эпизоде
            render_mode: Режим отрисовки ('human' или 'rgb_array')
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_obstacle = reward_obstacle
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Определяем препятствия
        self.obstacles = obstacles if obstacles is not None else []
        
        # Определяем начальную позицию и цель
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        
        # Генерируем или используем предзаданную карту цветов
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
        
        # Пространство наблюдений: индекс цвета текущей клетки (скаляр)
        self.observation_space = spaces.Box(
            low=0, high=self.num_colors - 1, shape=(1,), dtype=np.int32
        )
        
        # Внутреннее состояние
        self.agent_pos = None
        self.steps = 0
        
        # Для отрисовки
        self.fig = None
        self.ax = None
    
    def _get_obs(self) -> np.ndarray:
        """Возвращает текущее наблюдение (индекс цвета текущей клетки)."""
        color_idx = self.floor_colors[self.agent_pos[0], self.agent_pos[1]]
        return np.array([color_idx], dtype=np.int32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Возвращает дополнительную информацию о состоянии."""
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "steps": self.steps,
            "color_idx": self.floor_colors[self.agent_pos[0], self.agent_pos[1]],
            "distance_to_goal": np.abs(self.agent_pos[0] - self.goal_pos[0]) + 
                                np.abs(self.agent_pos[1] - self.goal_pos[1])
        }
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Сброс среды в начальное состояние."""
        super().reset(seed=seed)
        
        # Устанавливаем начальную позицию
        if self.start_pos is None:
            # Выбираем случайную позицию, избегая препятствий и цели
            while True:
                pos = self.np_random.integers(0, self.grid_size, size=2)
                if tuple(pos) not in self.obstacles:
                    if self.goal_pos is None or tuple(pos) != tuple(self.goal_pos):
                        self.agent_pos = tuple(pos)
                        break
        else:
            self.agent_pos = self.start_pos
        
        # Устанавливаем цель, если не задана
        if self.goal_pos is None:
            while True:
                goal = self.np_random.integers(0, self.grid_size, size=2)
                if tuple(goal) not in self.obstacles and tuple(goal) != self.agent_pos:
                    self.goal_pos = tuple(goal)
                    break
        
        self.steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Выполняет один шаг в среде."""
        # Движения: 0=вверх, 1=вправо, 2=вниз, 3=влево
        moves = {
            0: (-1, 0),  # вверх
            1: (0, 1),   # вправо
            2: (1, 0),   # вниз
            3: (0, -1),  # влево
        }
        
        # Вычисляем новую позицию
        move = moves[action]
        new_pos = (
            self.agent_pos[0] + move[0],
            self.agent_pos[1] + move[1]
        )
        
        # Проверяем границы
        if (new_pos[0] < 0 or new_pos[0] >= self.grid_size or
            new_pos[1] < 0 or new_pos[1] >= self.grid_size):
            # Выход за границы - остаемся на месте
            new_pos = self.agent_pos
            reward = self.reward_step
        # Проверяем препятствия
        elif new_pos in self.obstacles:
            # Столкновение с препятствием
            reward = self.reward_obstacle
            new_pos = self.agent_pos  # Остаемся на месте
        else:
            # Обычное движение
            self.agent_pos = new_pos
            reward = self.reward_step
        
        self.steps += 1
        
        # Проверяем, достигли ли цели
        terminated = self.agent_pos == self.goal_pos
        if terminated:
            reward = self.reward_goal
        
        # Эпизод завершается, если достигли цели или превысили max_steps
        truncated = self.steps >= self.max_steps
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render_rgb_array(self, agent_pos=None, goal_pos=None, obstacles=None):
        """Отрисовка в RGB массив для видео."""
        if agent_pos is None:
            agent_pos = self.agent_pos
        if goal_pos is None:
            goal_pos = self.goal_pos
        if obstacles is None:
            obstacles = self.obstacles
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Получаем палитру цветов
        color_palette = get_color_palette(self.num_colors)
        
        # Рисуем клетки с цветами
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in obstacles:
                    color_idx = self.floor_colors[i, j]
                    color = color_palette[color_idx % len(color_palette)]
                    rect = patches.Rectangle(
                        (j, i), 1, 1,
                        linewidth=1.5, edgecolor='black',
                        facecolor=color, alpha=0.8
                    )
                    ax.add_patch(rect)
                    # Добавляем номер цвета
                    if (i, j) != goal_pos:
                        brightness = sum(color) / 3
                        text_color = 'white' if brightness < 0.5 else 'black'
                        ax.text(j + 0.5, i + 0.5, str(color_idx), 
                               ha='center', va='center', fontsize=8, 
                               fontweight='bold', color=text_color, zorder=5)
        
        # Рисуем препятствия
        for obs in obstacles:
            rect = patches.Rectangle(
                (obs[1], obs[0]), 1, 1,
                linewidth=2, edgecolor='black',
                facecolor='gray', alpha=0.9
            )
            ax.add_patch(rect)
            # Крестик
            ax.plot([obs[1], obs[1] + 1], [obs[0], obs[0] + 1], 
                   'k-', linewidth=3, alpha=0.9, zorder=6)
            ax.plot([obs[1], obs[1] + 1], [obs[0] + 1, obs[0]], 
                   'k-', linewidth=3, alpha=0.9, zorder=6)
        
        # Рисуем цель
        goal_rect = patches.Rectangle(
            (goal_pos[1], goal_pos[0]), 1, 1,
            linewidth=3, edgecolor='green',
            facecolor='lightgreen', alpha=0.9
        )
        ax.add_patch(goal_rect)
        ax.text(goal_pos[1] + 0.5, goal_pos[0] + 0.5, 'G', 
               ha='center', va='center', fontsize=16, 
               fontweight='bold', color='darkgreen', zorder=7)
        
        # Рисуем агента
        agent_circle = patches.Circle(
            (agent_pos[1] + 0.5, agent_pos[0] + 0.5),
            0.3, color='blue', zorder=10
        )
        ax.add_patch(agent_circle)
        ax.text(agent_pos[1] + 0.5, agent_pos[0] + 0.5, 'A', 
               ha='center', va='center', fontsize=12, 
               fontweight='bold', color='white', zorder=11)
        
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'GridWorld Color - Step: {self.steps}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        
        return frame
    
    def render(self):
        """Отрисовка текущего состояния среды."""
        if self.render_mode is None:
            return
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()
        
        self.ax.clear()
        
        # Получаем палитру цветов
        color_palette = get_color_palette(self.num_colors)
        
        # Рисуем клетки с цветами
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in self.obstacles:
                    color_idx = self.floor_colors[i, j]
                    color = color_palette[color_idx % len(color_palette)]
                    rect = patches.Rectangle(
                        (j, i), 1, 1,
                        linewidth=1.5, edgecolor='black',
                        facecolor=color, alpha=0.8
                    )
                    self.ax.add_patch(rect)
                    # Добавляем номер цвета
                    if (i, j) != self.goal_pos:
                        brightness = sum(color) / 3
                        text_color = 'white' if brightness < 0.5 else 'black'
                        self.ax.text(j + 0.5, i + 0.5, str(color_idx), 
                                   ha='center', va='center', fontsize=8, 
                                   fontweight='bold', color=text_color, zorder=5)
        
        # Рисуем препятствия
        for obs in self.obstacles:
            rect = patches.Rectangle(
                (obs[1], obs[0]), 1, 1,
                linewidth=2, edgecolor='black',
                facecolor='gray', alpha=0.9
            )
            self.ax.add_patch(rect)
            # Крестик
            self.ax.plot([obs[1], obs[1] + 1], [obs[0], obs[0] + 1], 
                       'k-', linewidth=3, alpha=0.9, zorder=6)
            self.ax.plot([obs[1], obs[1] + 1], [obs[0] + 1, obs[0]], 
                       'k-', linewidth=3, alpha=0.9, zorder=6)
        
        # Рисуем цель
        goal_rect = patches.Rectangle(
            (self.goal_pos[1], self.goal_pos[0]), 1, 1,
            linewidth=3, edgecolor='green',
            facecolor='lightgreen', alpha=0.9
        )
        self.ax.add_patch(goal_rect)
        self.ax.text(self.goal_pos[1] + 0.5, self.goal_pos[0] + 0.5, 'G', 
                   ha='center', va='center', fontsize=16, 
                   fontweight='bold', color='darkgreen', zorder=7)
        
        # Рисуем агента
        agent_circle = patches.Circle(
            (self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5),
            0.3, color='blue', zorder=10
        )
        self.ax.add_patch(agent_circle)
        self.ax.text(self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5, 'A', 
                   ha='center', va='center', fontsize=12, 
                   fontweight='bold', color='white', zorder=11)
        
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.set_title(f'GridWorld Color - Step: {self.steps}', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def close(self):
        """Закрытие среды и освобождение ресурсов."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def get_shortest_path_length(self) -> int:
        """Вычисляет длину кратчайшего пути от старта до цели (BFS)."""
        from collections import deque
        
        if self.start_pos is None or self.goal_pos is None:
            return self.grid_size * 2  # Примерная оценка
        
        queue = deque([(self.start_pos, 0)])
        visited = {self.start_pos}
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while queue:
            pos, dist = queue.popleft()
            
            if pos == self.goal_pos:
                return dist
            
            for move in moves:
                new_pos = (pos[0] + move[0], pos[1] + move[1])
                
                if (0 <= new_pos[0] < self.grid_size and 
                    0 <= new_pos[1] < self.grid_size and
                    new_pos not in self.obstacles and
                    new_pos not in visited):
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        
        return self.grid_size * 2  # Если путь не найден, возвращаем оценку


def generate_task1_environments():
    """Генерирует PNG изображения для окружений из задания 1."""
    
    # Окружение 1: 5x5 с уникальными цветами для каждой позиции (25 цветов) без препятствий
    print("Генерация окружения 1: 5x5 с 25 уникальными цветами...")
    grid_size = 5
    start_pos = (0, 0)
    goal_pos = (4, 4)
    obstacles = []
    floor_colors = generate_floor_colors(grid_size, num_colors=None, seed=42)
    
    render_env_with_colors(
        grid_size, start_pos, goal_pos, obstacles, floor_colors,
        'task1_env1_5x5_25colors.png',
        'Окружение 1: 5x5, 25 уникальных цветов (MDP)',
        num_colors=25
    )
    
    # Окружение 2: 5x5 с 5 цветами пола без препятствий
    print("Генерация окружения 2: 5x5 с 5 цветами пола...")
    grid_size = 5
    start_pos = (0, 0)
    goal_pos = (4, 4)
    obstacles = []
    floor_colors = generate_floor_colors(grid_size, num_colors=5, seed=42)
    
    render_env_with_colors(
        grid_size, start_pos, goal_pos, obstacles, floor_colors,
        'task1_env2_5x5_5colors.png',
        'Окружение 2: 5x5, 5 цветов пола (POMDP)',
        num_colors=5
    )
    
    # Окружение 3: 10x10 с 7 цветами пола и 10% препятствий
    print("Генерация окружения 3: 10x10 с 7 цветами и 10% препятствий...")
    grid_size = 10
    start_pos = (0, 0)
    goal_pos = (9, 9)
    obstacles = generate_obstacles(
        grid_size, 
        obstacle_percentage=0.1, 
        seed=42,
        exclude_positions=[start_pos, goal_pos]
    )
    floor_colors = generate_floor_colors(grid_size, num_colors=7, seed=42)
    
    render_env_with_colors(
        grid_size, start_pos, goal_pos, obstacles, floor_colors,
        'task1_env3_10x10_7colors_10pct_obstacles.png',
        'Окружение 3: 10x10, 7 цветов пола, 10% препятствий',
        num_colors=7
    )
    
    # Окружение 4: 10x10 с 4 цветами пола и 10% препятствий
    print("Генерация окружения 4: 10x10 с 4 цветами и 10% препятствий...")
    grid_size = 10
    start_pos = (0, 0)
    goal_pos = (9, 9)
    obstacles = generate_obstacles(
        grid_size, 
        obstacle_percentage=0.1, 
        seed=42,
        exclude_positions=[start_pos, goal_pos]
    )
    floor_colors = generate_floor_colors(grid_size, num_colors=4, seed=42)
    
    render_env_with_colors(
        grid_size, start_pos, goal_pos, obstacles, floor_colors,
        'task1_env4_10x10_4colors_10pct_obstacles.png',
        'Окружение 4: 10x10, 4 цвета пола, 10% препятствий',
        num_colors=4
    )
    
    print("\nВсе изображения успешно созданы!")


# Пример использования
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # Генерация изображений окружений
        generate_task1_environments()
    else:
        # Тестирование среды
        env = GridWorldEnv(
            grid_size=5,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            obstacles=[(1, 1), (2, 2), (3, 3)],
            num_colors=5,
            render_mode="human"
        )
        
        # Тестируем среду
        obs, info = env.reset()
        print(f"Начальное наблюдение: {obs}")
        print(f"Информация: {info}")
        
        # Выполняем несколько случайных действий
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Шаг {step}: действие={action}, награда={reward:.2f}, "
                  f"наблюдение={obs}, завершено={terminated}")
            
            if terminated or truncated:
                print("Эпизод завершен!")
                break
        
        env.close()

