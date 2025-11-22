"""
Сравнительный анализ всех методов: DQN, DQN-LSTM, PPO, PPO-LSTM
Генерация одного красивого comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
from typing import Dict, List, Optional
import glob

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


def load_metrics_from_file(save_dir: str) -> Optional[Dict[str, List[float]]]:
    """Загрузить метрики из файла, если они были сохранены."""
    metrics_file = os.path.join(save_dir, 'metrics.pkl')
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Ошибка загрузки метрик из {metrics_file}: {e}")
    return None


def load_metrics_from_wandb(method: str, use_lstm: bool, env_id: int) -> Optional[Dict[str, List[float]]]:
    """Попытаться загрузить метрики из wandb (если доступно)."""
    try:
        import wandb
        api = wandb.Api()
        
        lstm_suffix = "lstm" if use_lstm else "no-lstm"
        run_name = f"{method}-{lstm_suffix}-env{env_id}"
        
        # Ищем последний run с таким именем
        runs = api.runs("sinitskii-mi/RL2", filters={"display_name": run_name})
        if runs:
            run = runs[0]
            # Извлекаем метрики из истории
            history = run.history()
            if not history.empty:
                metrics = {
                    'episode_rewards': history.get('episode_reward', []).tolist() if 'episode_reward' in history else [],
                    'episode_lengths': history.get('episode_length', []).tolist() if 'episode_length' in history else [],
                    'eval_rewards': history.get('eval_reward', []).tolist() if 'eval_reward' in history else [],
                    'eval_lengths': history.get('eval_length', []).tolist() if 'eval_length' in history else [],
                }
                if method == 'dqn':
                    metrics['losses'] = history.get('loss', []).tolist() if 'loss' in history else []
                else:  # ppo
                    metrics['policy_losses'] = history.get('policy_loss', []).tolist() if 'policy_loss' in history else []
                    metrics['value_losses'] = history.get('value_loss', []).tolist() if 'value_loss' in history else []
                return metrics
    except Exception as e:
        print(f"Не удалось загрузить из wandb: {e}")
    return None


def create_comparison_png(results_dir: str = "task2/results", output_path: str = "task2/results/comparisons/comparison.png"):
    """
    Создать один красивый comparison.png со всеми методами на всех окружениях.
    """
    methods = [
        ('dqn', False, 'DQN', '#1f77b4', '-', 2.5),  # синий
        ('dqn', True, 'DQN-LSTM', '#17becf', '--', 2.5),  # голубой
        ('ppo', False, 'PPO', '#d62728', '-', 2.5),  # красный
        ('ppo', True, 'PPO-LSTM', '#ff7f0e', '--', 2.5),  # оранжевый
    ]
    
    env_ids = [1, 2, 3]
    
    # Собираем все метрики
    all_data = {}
    for env_id in env_ids:
        all_data[env_id] = {}
        for method, use_lstm, label, color, linestyle, linewidth in methods:
            lstm_suffix = "lstm" if use_lstm else "no-lstm"
            save_dir = os.path.join(results_dir, f"{method}-{lstm_suffix}", f"env{env_id}")
            
            metrics = load_metrics_from_file(save_dir)
            if metrics is None:
                metrics = load_metrics_from_wandb(method, use_lstm, env_id)
            
            if metrics:
                all_data[env_id][label] = {
                    'metrics': metrics,
                    'color': color,
                    'linestyle': linestyle,
                    'linewidth': linewidth
                }
    
    # Создаем большой красивый график
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3, height_ratios=[1, 1, 1, 0.8])
    
    fig.suptitle('Сравнение всех методов на всех окружениях', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Для каждого окружения создаем график наград
    for env_idx, env_id in enumerate(env_ids):
        ax = fig.add_subplot(gs[0, env_idx])
        env_data = all_data.get(env_id, {})
        
        for label, data in env_data.items():
            metrics = data['metrics']
            color = data['color']
            linestyle = data['linestyle']
            linewidth = data['linewidth']
            
            if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 0:
                rewards = metrics['episode_rewards']
                window = min(50, len(rewards) // 10)
                if window > 1:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(rewards)), moving_avg, 
                           label=label, color=color, linestyle=linestyle, 
                           linewidth=linewidth, alpha=0.9)
                else:
                    ax.plot(rewards, label=label, color=color, linestyle=linestyle, 
                           linewidth=linewidth, alpha=0.7)
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Reward', fontweight='bold')
        ax.set_title(f'Окружение {env_id} - Episode Rewards', fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9, shadow=True, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # График длин эпизодов
    for env_idx, env_id in enumerate(env_ids):
        ax = fig.add_subplot(gs[1, env_idx])
        env_data = all_data.get(env_id, {})
        
        for label, data in env_data.items():
            metrics = data['metrics']
            color = data['color']
            linestyle = data['linestyle']
            linewidth = data['linewidth']
            
            if 'episode_lengths' in metrics and len(metrics['episode_lengths']) > 0:
                lengths = metrics['episode_lengths']
                window = min(50, len(lengths) // 10)
                if window > 1:
                    moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(lengths)), moving_avg, 
                           label=label, color=color, linestyle=linestyle, 
                           linewidth=linewidth, alpha=0.9)
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Length', fontweight='bold')
        ax.set_title(f'Окружение {env_id} - Episode Lengths', fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9, shadow=True, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # График Loss
    for env_idx, env_id in enumerate(env_ids):
        ax = fig.add_subplot(gs[2, env_idx])
        env_data = all_data.get(env_id, {})
        
        for label, data in env_data.items():
            metrics = data['metrics']
            color = data['color']
            linestyle = data['linestyle']
            linewidth = data['linewidth']
            
            if 'losses' in metrics and len(metrics['losses']) > 0:
                losses = metrics['losses']
                step = max(1, len(losses) // 1000)
                ax.plot(losses[::step], label=f'{label} Loss', color=color, 
                       linestyle=linestyle, linewidth=linewidth, alpha=0.8)
            elif 'policy_losses' in metrics and len(metrics['policy_losses']) > 0:
                policy_losses = metrics['policy_losses']
                step = max(1, len(policy_losses) // 1000)
                ax.plot(policy_losses[::step], label=f'{label} Policy', 
                       color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
        
        ax.set_xlabel('Update Step', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(f'Окружение {env_id} - Training Losses', fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9, shadow=True, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Финальное сравнение - барный график
    ax = fig.add_subplot(gs[3, :])
    
    x = np.arange(len(env_ids))
    width = 0.2
    
    summary_data = {}
    color_map = {}
    for method, use_lstm, label, color, _, _ in methods:
        summary_data[label] = []
        color_map[label] = color
        for env_id in env_ids:
            env_data = all_data.get(env_id, {})
            if label in env_data:
                metrics = env_data[label]['metrics']
                if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 0:
                    rewards = metrics['episode_rewards']
                    final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                    summary_data[label].append(final_avg)
                else:
                    summary_data[label].append(0)
            else:
                summary_data[label].append(0)
    
    for i, (label, values) in enumerate(summary_data.items()):
        color = color_map.get(label, 'gray')
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=label, alpha=0.8, 
                     color=color, edgecolor='black', linewidth=1.5)
        # Добавляем значения на столбцы
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Окружение', fontweight='bold', fontsize=14)
    ax.set_ylabel('Средняя награда (последние 100 эпизодов)', fontweight='bold', fontsize=14)
    ax.set_title('Финальное сравнение производительности', fontweight='bold', fontsize=16, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Окружение {i}' for i in env_ids], fontweight='bold', fontsize=12)
    ax.legend(loc='best', framealpha=0.9, shadow=True, fontsize=11, ncol=4)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Красивый comparison.png сохранен: {output_path}")
    plt.close()


def main():
    """Создать comparison.png."""
    print("="*60)
    print("Генерация comparison.png")
    print("="*60)
    
    create_comparison_png(output_path="results/comparisons/comparison.png")
    
    print("\n" + "="*60)
    print("Готово!")
    print("="*60)
    print(f"\nГрафик сохранен в: task2/results/comparisons/comparison.png")


if __name__ == "__main__":
    main()
