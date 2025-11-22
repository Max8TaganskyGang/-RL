"""
Скрипт для генерации PNG изображений окружений GridWorld
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gridworld_env import GridWorldEnv


def render_to_png(env: GridWorldEnv, filename: str, title: str = None):
    """
    Рендерит окружение в PNG файл.
    
    Args:
        env: Окружение GridWorld
        filename: Имя файла для сохранения
        title: Заголовок изображения
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Рисуем сетку
    for i in range(env.grid_size + 1):
        ax.axhline(i, color='black', linewidth=1.5)
        ax.axvline(i, color='black', linewidth=1.5)
    
    # Рисуем препятствия
    for obs in env.obstacles:
        rect = patches.Rectangle(
            (obs[1], obs[0]), 1, 1,
            linewidth=2, edgecolor='black',
            facecolor='gray', alpha=0.8
        )
        ax.add_patch(rect)
    
    # Рисуем цель
    goal_rect = patches.Rectangle(
        (env.goal_pos[1], env.goal_pos[0]), 1, 1,
        linewidth=3, edgecolor='green',
        facecolor='lightgreen', alpha=0.8
    )
    ax.add_patch(goal_rect)
    
    # Рисуем агента
    agent_circle = patches.Circle(
        (env.agent_pos[1] + 0.5, env.agent_pos[0] + 0.5),
        0.35, facecolor='blue', edgecolor='darkblue', 
        zorder=10, linewidth=2
    )
    ax.add_patch(agent_circle)
    
    # Добавляем метки
    ax.text(env.agent_pos[1] + 0.5, env.agent_pos[0] + 0.5, 'A', 
            ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    ax.text(env.goal_pos[1] + 0.5, env.goal_pos[0] + 0.5, 'G', 
            ha='center', va='center', fontsize=16, fontweight='bold', color='darkgreen')
    
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Чтобы (0,0) был вверху слева
    
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    else:
        ax.set_title(f'GridWorld {env.grid_size}x{env.grid_size}', 
                    fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Добавляем координаты
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if (i, j) not in env.obstacles and (i, j) != env.goal_pos and (i, j) != env.agent_pos:
                ax.text(j + 0.5, i + 0.5, f'({i},{j})', 
                       ha='center', va='center', fontsize=8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Сохранено: {filename}")


def main():
    """Генерирует PNG изображения для различных конфигураций GridWorld."""
    
    # Окружение 1: Простое 5x5 с препятствиями по диагонали
    print("Генерация окружения 1...")
    env1 = GridWorldEnv(
        grid_size=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[(1, 1), (2, 2), (3, 3)],
        render_mode=None
    )
    env1.reset()
    render_to_png(env1, 'env_1_simple.png', 'GridWorld: Простое окружение')
    
    # Окружение 2: 5x5 без препятствий
    print("Генерация окружения 2...")
    env2 = GridWorldEnv(
        grid_size=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[],
        render_mode=None
    )
    env2.reset()
    render_to_png(env2, 'env_2_no_obstacles.png', 'GridWorld: Без препятствий')
    
    # Окружение 3: 5x5 с препятствиями, образующими барьер
    print("Генерация окружения 3...")
    env3 = GridWorldEnv(
        grid_size=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[(1, 2), (2, 2), (3, 2)],
        render_mode=None
    )
    env3.reset()
    render_to_png(env3, 'env_3_barrier.png', 'GridWorld: С барьером')
    
    # Окружение 4: 5x5 с препятствиями в углах
    print("Генерация окружения 4...")
    env4 = GridWorldEnv(
        grid_size=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[(0, 1), (1, 0), (3, 4), (4, 3)],
        render_mode=None
    )
    env4.reset()
    render_to_png(env4, 'env_4_corners.png', 'GridWorld: Препятствия в углах')
    
    # Окружение 5: 4x4 простое
    print("Генерация окружения 5...")
    env5 = GridWorldEnv(
        grid_size=4,
        start_pos=(0, 0),
        goal_pos=(3, 3),
        obstacles=[(1, 1), (2, 2)],
        render_mode=None
    )
    env5.reset()
    render_to_png(env5, 'env_5_4x4.png', 'GridWorld: 4x4')
    
    # Окружение 6: 5x5 с лабиринтом
    print("Генерация окружения 6...")
    env6 = GridWorldEnv(
        grid_size=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[(0, 2), (1, 2), (2, 0), (2, 1), (2, 3), (2, 4), (3, 2)],
        render_mode=None
    )
    env6.reset()
    render_to_png(env6, 'env_6_maze.png', 'GridWorld: Лабиринт')
    
    print("\nВсе изображения успешно созданы!")


if __name__ == "__main__":
    main()

