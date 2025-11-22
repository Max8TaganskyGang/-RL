"""
Скрипт для сравнения обычной и векторизованных сред
Создает графики сравнения с учетом общего количества шагов
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from typing import Dict, List, Optional

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_metrics_from_wandb(run_name: str, project: str = "RL4") -> Optional[Dict]:
    """Загрузить метрики из Wandb."""
    try:
        import pandas as pd
        
        api = wandb.Api()
        runs = list(api.runs(f"sinitskii-mi/{project}", filters={"display_name": run_name}))
        
        if not runs:
            print(f"⚠️  Run не найден: {run_name}")
            return None
        
        # Берем последний завершенный run
        run = None
        for r in runs:
            if r.state == 'finished':
                run = r
                break
        if run is None:
            run = runs[0]
        
        print(f"📊 Загружаю данные из: {run.name} ({run.state})")
        
        # Загружаем историю
        history = run.history(pandas=True)
        
        if history.empty:
            print(f"⚠️  Пустая история для {run_name}")
            return None
        
        # Извлекаем данные
        metrics = {}
        
        # Episode rewards - берем все значения episode_reward
        if 'episode_reward' in history.columns:
            rewards = history['episode_reward'].dropna().tolist()
            metrics['episode_rewards'] = rewards
        else:
            metrics['episode_rewards'] = []
        
        # Episode lengths
        if 'episode_length' in history.columns:
            lengths = history['episode_length'].dropna().tolist()
            metrics['episode_lengths'] = lengths
        else:
            metrics['episode_lengths'] = []
        
        # Total steps - берем последнее значение из истории
        if 'total_steps' in history.columns:
            total_steps_values = history['total_steps'].dropna().values
            if len(total_steps_values) > 0:
                metrics['total_steps'] = float(total_steps_values[-1])
            else:
                metrics['total_steps'] = None
        else:
            # Если нет total_steps, вычисляем как сумму длин эпизодов
            if metrics.get('episode_lengths'):
                metrics['total_steps'] = sum(metrics['episode_lengths'])
            else:
                metrics['total_steps'] = None
        
        # Losses
        if 'loss' in history.columns:
            metrics['losses'] = history['loss'].dropna().tolist()
        else:
            metrics['losses'] = []
        
        print(f"   Загружено: {len(metrics.get('episode_rewards', []))} эпизодов")
        if metrics.get('total_steps'):
            print(f"   Всего шагов: {metrics['total_steps']:.0f}")
        
        return metrics
    except Exception as e:
        print(f"⚠️  Не удалось загрузить из wandb: {e}")
        import traceback
        traceback.print_exc()
    return None


def create_comparison_graph(
    env_id: int = 1,
    project: str = "RL4",
    save_path: str = "task4/results/comparison.png"
):
    """Создать график сравнения обычной и векторизованных сред."""
    try:
        import pandas as pd
    except ImportError:
        pd = None
    
    print("="*60)
    print(f"Генерация графика сравнения для окружения {env_id}")
    print("="*60)
    
    # Загружаем данные из Wandb
    methods = {
        'normal': ('Обычная', '#1f77b4', '-', 'normal-env1'),
        'sync': ('SyncVectorEnv', '#ff7f0e', '--', f'sync-vectorized-env{env_id}'),
        'full': ('Полная векторизация', '#2ca02c', ':', f'full-vectorized-env{env_id}'),
    }
    
    all_data = {}
    times = {}
    total_steps_dict = {}
    
    for key, (label, color, linestyle, run_name) in methods.items():
        metrics = load_metrics_from_wandb(run_name, project=project)
        if metrics:
            all_data[key] = {
                'metrics': metrics,
                'label': label,
                'color': color,
                'linestyle': linestyle,
            }
            
            # Получаем время из summary run (если доступно)
            try:
                api = wandb.Api()
                runs = list(api.runs(f"sinitskii-mi/{project}", filters={"display_name": run_name}))
                if runs:
                    run = runs[0]
                    # Время можно получить из _wandb или вычислить
                    # Для упрощения используем None
                    times[key] = None
            except:
                times[key] = None
    
    if not all_data:
        print("⚠️  Нет данных для сравнения!")
        return
    
    # Создаем график
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Сравнение обычной и векторизованных сред (Окружение {env_id})', 
                 fontsize=18, fontweight='bold', y=0.99)
    
    # График 1: Episode Rewards vs Total Steps
    ax1 = fig.add_subplot(gs[0, 0])
    for key, data in all_data.items():
        metrics = data['metrics']
        rewards = metrics.get('episode_rewards', [])
        total_steps = metrics.get('total_steps')
        
        if rewards:
            # Если есть total_steps для каждого эпизода, используем их
            if 'total_steps_list' in metrics:
                steps_list = metrics['total_steps_list']
            else:
                # Иначе создаем кумулятивную сумму длин эпизодов
                if key == 'normal':
                    # Для обычной среды: каждый эпизод добавляет свою длину
                    steps_list = np.cumsum(metrics.get('episode_lengths', []))
                else:
                    # Для векторизованных: каждый эпизод добавляет длину * num_envs
                    # Но нам нужно общее количество шагов по всем средам
                    lengths = metrics.get('episode_lengths', [])
                    # Предполагаем num_envs = 4 (можно получить из config)
                    steps_list = np.cumsum([l * 4 for l in lengths])  # Упрощение
            
            window = min(10, len(rewards) // 10)
            if window > 1 and len(rewards) > window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                steps_avg = steps_list[window-1:]
                ax1.plot(steps_avg, moving_avg, 
                        label=data['label'], color=data['color'], 
                        linestyle=data['linestyle'], linewidth=2.5, alpha=0.8)
            else:
                ax1.plot(steps_list[:len(rewards)], rewards, 
                        label=data['label'], color=data['color'], 
                        linestyle=data['linestyle'], linewidth=2.5, alpha=0.8)
    
    ax1.set_xlabel('Общее количество шагов (сумма по всем средам)', fontweight='bold')
    ax1.set_ylabel('Награда за эпизод', fontweight='bold')
    ax1.set_title('Episode Rewards vs Total Steps', fontweight='bold', pad=10)
    ax1.legend(loc='best', framealpha=0.9, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # График 2: Episode Lengths vs Total Steps
    ax2 = fig.add_subplot(gs[0, 1])
    for key, data in all_data.items():
        metrics = data['metrics']
        lengths = metrics.get('episode_lengths', [])
        
        if lengths:
            if key == 'normal':
                steps_list = np.cumsum(lengths)
            else:
                steps_list = np.cumsum([l * 4 for l in lengths])  # Упрощение
            
            window = min(10, len(lengths) // 10)
            if window > 1 and len(lengths) > window:
                moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
                steps_avg = steps_list[window-1:]
                ax2.plot(steps_avg, moving_avg, 
                        label=data['label'], color=data['color'], 
                        linestyle=data['linestyle'], linewidth=2.5, alpha=0.8)
            else:
                ax2.plot(steps_list[:len(lengths)], lengths, 
                        label=data['label'], color=data['color'], 
                        linestyle=data['linestyle'], linewidth=2.5, alpha=0.8)
    
    ax2.set_xlabel('Общее количество шагов (сумма по всем средам)', fontweight='bold')
    ax2.set_ylabel('Длина эпизода', fontweight='bold')
    ax2.set_title('Episode Lengths vs Total Steps', fontweight='bold', pad=10)
    ax2.legend(loc='best', framealpha=0.9, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # График 3: Episode Rewards vs Episode Number
    ax3 = fig.add_subplot(gs[1, 0])
    for key, data in all_data.items():
        metrics = data['metrics']
        rewards = metrics.get('episode_rewards', [])
        
        if rewards:
            episodes = list(range(len(rewards)))
            window = min(10, len(rewards) // 10)
            if window > 1 and len(rewards) > window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax3.plot(episodes[window-1:], moving_avg, 
                        label=data['label'], color=data['color'], 
                        linestyle=data['linestyle'], linewidth=2.5, alpha=0.8)
            else:
                ax3.plot(episodes, rewards, 
                        label=data['label'], color=data['color'], 
                        linestyle=data['linestyle'], linewidth=2.5, alpha=0.8)
    
    ax3.set_xlabel('Номер эпизода', fontweight='bold')
    ax3.set_ylabel('Награда за эпизод', fontweight='bold')
    ax3.set_title('Episode Rewards vs Episode Number', fontweight='bold', pad=10)
    ax3.legend(loc='best', framealpha=0.9, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # График 4: Таблица сравнения скорости
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Создаем таблицу с результатами
    table_data = []
    headers = ['Метод', 'Всего шагов', 'Финальная награда']
    
    for key, data in all_data.items():
        metrics = data['metrics']
        total_steps = metrics.get('total_steps')
        final_reward = np.mean(metrics.get('episode_rewards', [])[-100:]) if len(metrics.get('episode_rewards', [])) >= 100 else np.mean(metrics.get('episode_rewards', [])) if metrics.get('episode_rewards') else 0
        
        steps_str = f"{total_steps:.0f}" if total_steps else "N/A"
        reward_str = f"{final_reward:.2f}" if final_reward else "N/A"
        
        table_data.append([data['label'], steps_str, reward_str])
    
    table = ax4.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Стилизация таблицы
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Сравнение результатов', fontweight='bold', pad=20)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ График сохранен: {save_path}")
    plt.close()


if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  pandas не установлен, некоторые функции могут не работать")
        pd = None
    
    import argparse
    parser = argparse.ArgumentParser(description='Сравнение векторизованных сред')
    parser.add_argument('--env', type=int, default=1, help='ID окружения')
    parser.add_argument('--project', type=str, default='RL4', help='Wandb проект')
    parser.add_argument('--output', type=str, default='task4/results/comparison.png', help='Путь для сохранения')
    
    args = parser.parse_args()
    
    create_comparison_graph(env_id=args.env, project=args.project, save_path=args.output)

