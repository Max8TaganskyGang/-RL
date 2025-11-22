"""
Генерация изображений окружений для Task 3 (MNIST)
Показывает окружения, где каждая клетка отображает MNIST изображение соответствующей цифры
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridworld_env import generate_floor_colors, generate_obstacles
from task3.gridworld_mnist_env import GridWorldMNISTEnv


def render_mnist_env(env, filename, title):
    """
    Рендерит окружение, где каждая клетка показывает MNIST изображение соответствующей цифры.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlim(-0.5, env.grid_size + 0.5)
    ax.set_ylim(-0.5, env.grid_size + 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Рисуем сетку
    for i in range(env.grid_size + 1):
        ax.axhline(i, color='black', linewidth=2)
        ax.axvline(i, color='black', linewidth=2)
    
    # Для каждой клетки показываем MNIST изображение
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            pos = (i, j)
            
            if pos in env.obstacles:
                # Препятствие - черная клетка с крестом
                rect = patches.Rectangle(
                    (j, i), 1, 1,
                    linewidth=2, edgecolor='red',
                    facecolor='black', alpha=0.9
                )
                ax.add_patch(rect)
                ax.text(j + 0.5, i + 0.5, '✕', 
                       ha='center', va='center', 
                       fontsize=24, color='red', fontweight='bold')
            else:
                # Получаем цвет (индекс) этой клетки
                color_idx = env.floor_colors[i, j]
                digit = color_idx % 10
                
                # Получаем пример MNIST изображения для этой цифры
                available_images = env.mnist_data[digit]
                if len(available_images) > 0:
                    # Берем первое изображение
                    mnist_img = available_images[0]
                    
                    # Используем imshow для отображения изображения на всю клетку
                    # extent: [left, right, bottom, top]
                    # С учетом invert_yaxis: bottom > top для правильной ориентации
                    extent = [j, j + 1, i + 1, i]
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
                    
                    # Показываем номер цвета маленьким текстом в углу
                    ax.text(j + 0.1, i + 0.1, str(color_idx),
                           ha='left', va='bottom',
                           fontsize=10, fontweight='bold',
                           color='red', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # Старт - зеленая рамка
    start_rect = patches.Rectangle(
        (env.start_pos[1], env.start_pos[0]), 1, 1,
        linewidth=4, edgecolor='green',
        facecolor='none', zorder=10
    )
    ax.add_patch(start_rect)
    ax.text(env.start_pos[1] + 0.5, env.start_pos[0] - 0.2, 'START',
           ha='center', va='top', fontsize=12, 
           fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Цель - красная рамка
    goal_rect = patches.Rectangle(
        (env.goal_pos[1], env.goal_pos[0]), 1, 1,
        linewidth=4, edgecolor='red',
        facecolor='none', zorder=10
    )
    ax.add_patch(goal_rect)
    ax.text(env.goal_pos[1] + 0.5, env.goal_pos[0] + 1.2, 'GOAL',
           ha='center', va='bottom', fontsize=12,
           fontweight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Column', fontweight='bold', fontsize=12)
    ax.set_ylabel('Row', fontweight='bold', fontsize=12)
    
    # Легенда
    legend_text = "Легенда:\n"
    legend_text += "• Каждая клетка показывает MNIST изображение цифры (0-9)\n"
    legend_text += "• Цифра соответствует индексу цвета (показан в углу клетки)\n"
    legend_text += "• START - Стартовая позиция (зеленая рамка)\n"
    legend_text += "• GOAL - Целевая позиция (красная рамка)\n"
    legend_text += "• ✕ - Препятствие/опасность (черная клетка с красным крестом)\n"
    legend_text += "• Агент получает случайное MNIST изображение соответствующей цифры как наблюдение"
    
    fig.text(0.02, 0.02, legend_text, fontsize=10, 
            verticalalignment='bottom', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Сохранено: {filename}")
    plt.close()


def generate_task3_environments():
    """Генерирует изображения всех окружений для Task 3."""
    output_dir = "task3/task3_photos"
    os.makedirs(output_dir, exist_ok=True)
    
    seed = 42
    
    # Окружение 1: 3x3 с уникальным числом на каждую клетку (9 цветов)
    print("\nГенерация окружения 1...")
    grid_size1 = 3
    goal_pos1 = (2, 2)
    # Создаем уникальные цвета для каждой клетки (0-8)
    floor_colors1 = np.zeros((grid_size1, grid_size1), dtype=int)
    color_idx = 0
    for i in range(grid_size1):
        for j in range(grid_size1):
            if (i, j) != goal_pos1:
                color_idx += 1
                floor_colors1[i, j] = color_idx
    # Цвет 0 только у цели
    floor_colors1[goal_pos1[0], goal_pos1[1]] = 0
    
    env1 = GridWorldMNISTEnv(
        grid_size=grid_size1,
        start_pos=(0, 0),
        goal_pos=goal_pos1,
        obstacles=[],
        floor_colors=floor_colors1,
        seed=seed,
        image_size=28,
        flatten=False
    )
    render_mnist_env(
        env1,
        f"{output_dir}/task3_env1_3x3_9colors_unique_mnist.png",
        "Окружение 1: 3x3, уникальное число на каждую клетку (0-8)\n(Каждая клетка = MNIST изображение цифры)"
    )
    env1.close()
    
    # Окружение 2: 10x10 с 7 цветами пола и 10% препятствий
    print("\nГенерация окружения 2...")
    goal_pos2 = (9, 9)
    obstacles2 = generate_obstacles(
        10, 
        obstacle_percentage=0.1, 
        seed=seed,
        exclude_positions=[(0, 0), goal_pos2]
    )
    floor_colors2 = generate_floor_colors(10, num_colors=7, seed=seed)
    # Цвет 0 только у цели
    floor_colors2[goal_pos2[0], goal_pos2[1]] = 0
    # Остальные цвета от 1 до 6
    for i in range(10):
        for j in range(10):
            if (i, j) != goal_pos2 and (i, j) not in obstacles2:
                if floor_colors2[i, j] == 0:
                    floor_colors2[i, j] = 1
    
    env2 = GridWorldMNISTEnv(
        grid_size=10,
        start_pos=(0, 0),
        goal_pos=goal_pos2,
        obstacles=obstacles2,
        floor_colors=floor_colors2,
        seed=seed,
        image_size=28,
        flatten=False
    )
    render_mnist_env(
        env2,
        f"{output_dir}/task3_env2_10x10_7colors_10pct_obstacles_mnist.png",
        "Окружение 2: 10x10, 7 цветов пола, 10% препятствий\n(Каждая клетка = MNIST изображение цифры)"
    )
    env2.close()
    
    # Окружение 3: 10x10 с 4 цветами пола и 10% препятствий
    print("\nГенерация окружения 3...")
    goal_pos3 = (9, 9)
    obstacles3 = generate_obstacles(
        10, 
        obstacle_percentage=0.1, 
        seed=seed,
        exclude_positions=[(0, 0), goal_pos3]
    )
    floor_colors3 = generate_floor_colors(10, num_colors=4, seed=seed)
    # Цвет 0 только у цели
    floor_colors3[goal_pos3[0], goal_pos3[1]] = 0
    # Остальные цвета от 1 до 3
    for i in range(10):
        for j in range(10):
            if (i, j) != goal_pos3 and (i, j) not in obstacles3:
                if floor_colors3[i, j] == 0:
                    floor_colors3[i, j] = 1
    
    env3 = GridWorldMNISTEnv(
        grid_size=10,
        start_pos=(0, 0),
        goal_pos=goal_pos3,
        obstacles=obstacles3,
        floor_colors=floor_colors3,
        seed=seed,
        image_size=28,
        flatten=False
    )
    render_mnist_env(
        env3,
        f"{output_dir}/task3_env3_10x10_4colors_10pct_obstacles_mnist.png",
        "Окружение 3: 10x10, 4 цвета пола, 10% препятствий\n(Каждая клетка = MNIST изображение цифры)"
    )
    env3.close()
    
    print("\n✅ Все изображения окружений Task 3 успешно созданы!")
    print("Теперь каждая клетка показывает MNIST изображение соответствующей цифры!")


if __name__ == "__main__":
    generate_task3_environments()
