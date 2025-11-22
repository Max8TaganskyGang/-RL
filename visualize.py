"""
Визуализация результатов обучения
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available. Install with: pip install imageio imageio-ffmpeg")


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves"
):
    """
    Построить графики обучения.
    
    Args:
        metrics: Словарь с метриками
        save_path: Путь для сохранения графика
        title: Заголовок графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # График наград
    ax = axes[0, 0]
    if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 0:
        rewards = metrics['episode_rewards']
        ax.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        # Скользящее среднее
        if len(rewards) > 10:
            window = min(100, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 
                   color='red', linewidth=2, label=f'Moving Avg ({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # График длин эпизодов
    ax = axes[0, 1]
    if 'episode_lengths' in metrics and len(metrics['episode_lengths']) > 0:
        lengths = metrics['episode_lengths']
        ax.plot(lengths, alpha=0.3, color='green', label='Episode Length')
        
        # Скользящее среднее
        if len(lengths) > 10:
            window = min(100, len(lengths) // 10)
            moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(lengths)), moving_avg, 
                   color='orange', linewidth=2, label=f'Moving Avg ({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # График loss (для DQN)
    ax = axes[1, 0]
    if 'losses' in metrics and len(metrics['losses']) > 0:
        losses = metrics['losses']
        ax.plot(losses, color='purple', alpha=0.7)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (DQN)')
        ax.grid(True, alpha=0.3)
    elif 'policy_losses' in metrics and len(metrics['policy_losses']) > 0:
        # Для PPO
        policy_losses = metrics['policy_losses']
        value_losses = metrics['value_losses']
        
        ax.plot(policy_losses, alpha=0.7, label='Policy Loss', color='blue')
        ax.plot(value_losses, alpha=0.7, label='Value Loss', color='red')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses (PPO)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # График оценки
    ax = axes[1, 1]
    if 'eval_rewards' in metrics and len(metrics['eval_rewards']) > 0:
        eval_rewards = metrics['eval_rewards']
        eval_lengths = metrics['eval_lengths']
        
        ax2 = ax.twinx()
        line1 = ax.plot(eval_rewards, 'o-', color='blue', label='Eval Reward', linewidth=2)
        line2 = ax2.plot(eval_lengths, 's-', color='red', label='Eval Length', linewidth=2)
        
        ax.set_xlabel('Evaluation')
        ax.set_ylabel('Reward', color='blue')
        ax2.set_ylabel('Length', color='red')
        ax.set_title('Evaluation Results')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_episode(env, agent, method: str = 'dqn', save_path: Optional[str] = None):
    """
    Визуализировать один эпизод работы обученного агента.
    
    Args:
        env: Окружение
        agent: Обученный агент
        method: Метод ('dqn' или 'ppo')
        save_path: Путь для сохранения
    """
    obs, info = env.reset()
    trajectory = [obs.copy()]
    actions_taken = []
    
    done = False
    step = 0
    max_steps = 200
    
    while not done and step < max_steps:
        if method == 'dqn':
            action = agent.select_action(obs, training=False)
        else:  # ppo
            action, _, _ = agent.select_action(obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        trajectory.append(obs.copy())
        actions_taken.append(action)
        step += 1
    
    # Визуализация траектории
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Рисуем сетку
    grid_size = env.grid_size
    for i in range(grid_size + 1):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)
    
    # Рисуем препятствия
    for obs_pos in env.obstacles:
        rect = plt.Rectangle((obs_pos[1], obs_pos[0]), 1, 1,
                           linewidth=1, edgecolor='black',
                           facecolor='gray', alpha=0.7)
        ax.add_patch(rect)
    
    # Рисуем цель
    goal_rect = plt.Rectangle((env.goal_pos[1], env.goal_pos[0]), 1, 1,
                            linewidth=2, edgecolor='green',
                            facecolor='lightgreen', alpha=0.7)
    ax.add_patch(goal_rect)
    
    # Рисуем траекторию
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 1] + 0.5, trajectory[:, 0] + 0.5, 
           'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax.scatter(trajectory[0, 1] + 0.5, trajectory[0, 0] + 0.5,
              color='blue', s=100, marker='o', label='Start', zorder=10)
    ax.scatter(trajectory[-1, 1] + 0.5, trajectory[-1, 0] + 0.5,
              color='red', s=100, marker='x', label='End', zorder=10)
    
    # Стрелки для направления движения
    for i in range(len(trajectory) - 1):
        dx = trajectory[i+1, 1] - trajectory[i, 1]
        dy = trajectory[i+1, 0] - trajectory[i, 0]
        if dx != 0 or dy != 0:
            ax.arrow(trajectory[i, 1] + 0.5, trajectory[i, 0] + 0.5,
                    dx * 0.3, dy * 0.3,
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f'Agent Trajectory ({method.upper()}) - {len(trajectory)-1} steps')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Траектория сохранена: {save_path}")
    else:
        plt.show()
    
    plt.close()


def record_episode_video(env, agent, method: str = 'dqn', save_path: Optional[str] = None, 
                        fps: int = 2, max_steps: int = 200):
    """
    Записать видео эпизода работы агента.
    
    Args:
        env: Окружение
        agent: Обученный агент
        method: Метод ('dqn' или 'ppo')
        save_path: Путь для сохранения видео
        fps: FPS видео
        max_steps: Максимальное количество шагов
    """
    if not IMAGEIO_AVAILABLE:
        print("Error: imageio not available. Cannot record video.")
        return
    
    obs, info = env.reset()
    frames = []
    trajectory = []
    
    done = False
    step = 0
    
    # Создаем фигуру для рендеринга
    fig, ax = plt.subplots(figsize=(10, 10))
    
    while not done and step < max_steps:
        # Используем render_rgb_array из среды для правильной отрисовки цветов
        if hasattr(env, 'render_rgb_array'):
            frame = env.render_rgb_array()
        else:
            # Fallback на старый способ
            ax.clear()
            grid_size = env.grid_size
            for i in range(grid_size + 1):
                ax.axhline(i, color='black', linewidth=1.5, alpha=0.5)
                ax.axvline(i, color='black', linewidth=1.5, alpha=0.5)
            
            for obs_pos in env.obstacles:
                rect = plt.Rectangle((obs_pos[1], obs_pos[0]), 1, 1,
                                   linewidth=2, edgecolor='black',
                                   facecolor='gray', alpha=0.9)
                ax.add_patch(rect)
            
            goal_rect = plt.Rectangle((env.goal_pos[1], env.goal_pos[0]), 1, 1,
                                    linewidth=3, edgecolor='green',
                                    facecolor='lightgreen', alpha=0.9)
            ax.add_patch(goal_rect)
            
            agent_circle = plt.Circle((env.agent_pos[1] + 0.5, env.agent_pos[0] + 0.5),
                                    0.35, facecolor='blue', edgecolor='darkblue', 
                                    zorder=10, linewidth=2)
            ax.add_patch(agent_circle)
            
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            plt.tight_layout()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3]
        
        frames.append(frame.copy())
        
        # Выбираем действие
        if method == 'dqn':
            # Для DQN с LSTM нужно сбросить hidden state в начале эпизода
            if step == 0 and hasattr(agent, 'use_lstm') and agent.use_lstm:
                action = agent.select_action(obs, training=False, reset_hidden=True)
            else:
                action = agent.select_action(obs, training=False)
        else:  # ppo
            # Для PPO с LSTM нужно сбросить hidden state в начале эпизода
            if step == 0 and hasattr(agent, 'use_lstm') and agent.use_lstm:
                action, _, _ = agent.select_action(obs, reset_hidden=True)
            else:
                action, _, _ = agent.select_action(obs)
        
        # Выполняем действие
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        trajectory.append(env.agent_pos)
        step += 1
    
    # Закрываем фигуру после записи (если она была создана)
    if 'fig' in locals():
        plt.close(fig)
    
    # Сохраняем видео
    if save_path and frames:
        imageio.mimsave(save_path, frames, fps=fps)
        print(f"Видео сохранено: {save_path} ({len(frames)} кадров)")
    elif not frames:
        print("Warning: Нет кадров для сохранения видео")

