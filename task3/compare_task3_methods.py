"""
Сравнительный анализ методов для Task 3 (MNIST)
Генерация comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from typing import Dict, List, Optional
import pandas as pd

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_metrics_from_file(save_dir: str) -> Optional[Dict[str, List[float]]]:
    """Загрузить метрики из файла."""
    metrics_file = os.path.join(save_dir, 'metrics.pkl')
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Ошибка загрузки метрик из {metrics_file}: {e}")
    return None


def load_metrics_from_wandb(method: str, env_id: int) -> Optional[Dict[str, List[float]]]:
    """Загрузить метрики из wandb."""
    try:
        import wandb
        api = wandb.Api()
        
        run_name = f"{method}-cnn-env{env_id}"
        
        # Ищем последний завершенный run с таким именем
        runs = api.runs("sinitskii-mi/RL3", filters={"display_name": run_name})
        if not runs:
            print(f"⚠️  Run не найден: {run_name}")
            return None
        
        # Берем последний завершенный run или последний вообще
        run = None
        for r in runs:
            if r.state == 'finished':
                run = r
                break
        if run is None:
            run = runs[0]  # Берем последний, даже если не завершен
        
        print(f"📊 Загружаю данные из: {run.name} ({run.state})")
        
        # Загружаем историю как pandas DataFrame для удобства
        history_df = run.history(pandas=True)
        
        if history_df.empty:
            print(f"⚠️  Пустая история для {run_name}")
            return None
        
        # Извлекаем данные напрямую из DataFrame
        episode_rewards = []
        episode_lengths = []
        eval_rewards = []
        eval_lengths = []
        losses = []
        policy_losses = []
        value_losses = []
        
        # Если есть колонка episode, используем её, иначе используем индекс
        if 'episode' in history_df.columns:
            # Удаляем строки где episode = NaN перед сортировкой
            history_df = history_df.dropna(subset=['episode']).sort_values('episode')
            # Конвертируем в int, пропуская NaN
            episode_col = history_df['episode'].fillna(0).astype(int).values
        else:
            # Создаем episode из индекса
            episode_col = history_df.index.values
        
        # Собираем все данные
        for idx in range(len(history_df)):
            ep = int(episode_col[idx]) if idx < len(episode_col) else idx
            
            if 'episode_reward' in history_df.columns:
                val = history_df.iloc[idx]['episode_reward']
                if pd.notna(val):
                    episode_rewards.append((ep, float(val)))
            
            if 'episode_length' in history_df.columns:
                val = history_df.iloc[idx]['episode_length']
                if pd.notna(val):
                    episode_lengths.append((ep, float(val)))
            
            if 'eval_reward' in history_df.columns:
                val = history_df.iloc[idx]['eval_reward']
                if pd.notna(val):
                    eval_rewards.append((ep, float(val)))
            
            if 'eval_length' in history_df.columns:
                val = history_df.iloc[idx]['eval_length']
                if pd.notna(val):
                    eval_lengths.append((ep, float(val)))
            
            if 'loss' in history_df.columns:
                val = history_df.iloc[idx]['loss']
                if pd.notna(val):
                    losses.append((ep, float(val)))
            
            if 'policy_loss' in history_df.columns:
                val = history_df.iloc[idx]['policy_loss']
                if pd.notna(val):
                    policy_losses.append((ep, float(val)))
            
            if 'value_loss' in history_df.columns:
                val = history_df.iloc[idx]['value_loss']
                if pd.notna(val):
                    value_losses.append((ep, float(val)))
        
        # Создаем упорядоченные списки (сортируем по эпизоду)
        episode_rewards.sort(key=lambda x: x[0])
        episode_lengths.sort(key=lambda x: x[0])
        
        # Извлекаем только значения в правильном порядке
        if episode_rewards:
            # Находим минимальный и максимальный эпизод
            min_ep = int(episode_rewards[0][0])
            max_ep = int(episode_rewards[-1][0])
            
            # Создаем словари для быстрого доступа
            rewards_dict = {int(ep): float(val) for ep, val in episode_rewards}
            lengths_dict = {int(ep): float(val) for ep, val in episode_lengths}
            
            # Создаем последовательные списки (начиная с min_ep)
            rewards_clean = []
            lengths_clean = []
            episode_indices = []  # Сохраняем номера эпизодов для правильной отрисовки
            
            # Заполняем списки для всех эпизодов от min_ep до max_ep
            for ep in range(min_ep, max_ep + 1):
                episode_indices.append(ep)
                if ep in rewards_dict:
                    rewards_clean.append(rewards_dict[ep])
                elif len(rewards_clean) > 0:  # Интерполяция для пропусков
                    rewards_clean.append(rewards_clean[-1])
                else:  # Если это первый эпизод и данных нет, ставим 0
                    rewards_clean.append(0.0)
                
                if ep in lengths_dict:
                    lengths_clean.append(lengths_dict[ep])
                elif len(lengths_clean) > 0:  # Интерполяция для пропусков
                    lengths_clean.append(lengths_clean[-1])
                else:  # Если это первый эпизод и данных нет, ставим 0
                    lengths_clean.append(0.0)
            
            # Нормализуем индексы: начинаем с 0 для графика, но сохраняем оригинальные номера
            # Сохраняем минимальный эпизод для правильной подписи осей
            if min_ep > 0:
                # Если эпизоды начинаются не с 0, создаем смещение
                # Но на графике будем показывать правильные номера через episode_indices
                pass  # episode_indices уже содержит правильные номера
        else:
            # Если нет данных по эпизодам, создаем пустые списки
            rewards_clean = []
            lengths_clean = []
            episode_indices = []
        
        # Нормализуем индексы: если min_ep > 0, то начинаем график с 0, но используем правильные номера
        if episode_rewards and len(episode_indices) > 0:
            min_ep_actual = min(episode_indices)
            # Если эпизоды начинаются не с 0, создаем смещение для нормализации
            # Но сохраняем оригинальные индексы для информации
            episode_offset = min_ep_actual if min_ep_actual > 0 else 0
        else:
            episode_offset = 0
        
        metrics = {
            'episode_rewards': rewards_clean if rewards_clean else [],
            'episode_lengths': lengths_clean if lengths_clean else [],
            'episode_indices': episode_indices if 'episode_indices' in locals() else [],
            'episode_offset': episode_offset,
            'eval_rewards': [v for _, v in eval_rewards] if eval_rewards else [],
            'eval_lengths': [v for _, v in eval_lengths] if eval_lengths else [],
        }
        
        if method == 'dqn':
            metrics['losses'] = [v for _, v in losses] if losses else []
        else:  # ppo
            metrics['policy_losses'] = [v for _, v in policy_losses] if policy_losses else []
            metrics['value_losses'] = [v for _, v in value_losses] if value_losses else []
        
        print(f"   Загружено: {len(rewards_clean)} эпизодов наград (эпизоды {episode_indices[0] if episode_indices else 'N/A'}-{episode_indices[-1] if episode_indices else 'N/A'}), {len(lengths_clean)} эпизодов длин")
        
        return metrics
    except Exception as e:
        print(f"⚠️  Не удалось загрузить из wandb: {e}")
        import traceback
        traceback.print_exc()
    return None


def create_comparison_png(results_dir: str = "task3/results", output_path: str = "task3/results/comparison.png"):
    """Создать comparison.png для Task 3."""
    methods = [
        ('dqn', 'DQN', '#1f77b4', '-', 3.0),  # Синяя сплошная, толще
        ('ppo', 'PPO', '#d62728', '--', 3.0),  # Красная пунктирная, толще
    ]
    
    env_ids = [1, 2, 3]
    
    # Собираем все метрики
    all_data = {}
    for env_id in env_ids:
        all_data[env_id] = {}
        for method, label, color, linestyle, linewidth in methods:
            save_dir = os.path.join(results_dir, "task3", f"{method}-cnn", f"env{env_id}")
            
            metrics = load_metrics_from_file(save_dir)
            if metrics is None:
                metrics = load_metrics_from_wandb(method, env_id)
            
            if metrics:
                all_data[env_id][label] = {
                    'metrics': metrics,
                    'color': color,
                    'linestyle': linestyle,
                    'linewidth': linewidth
                }
    
    # Создаем большой красивый график
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Сравнение DQN и PPO на Task 3 (MNIST наблюдения)', 
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
                # Используем правильные индексы эпизодов (начиная с реального первого эпизода)
                if 'episode_indices' in metrics and len(metrics['episode_indices']) == len(rewards):
                    episodes = metrics['episode_indices']
                else:
                    # Если нет сохраненных индексов, создаем от 0
                    offset = metrics.get('episode_offset', 0)
                    episodes = list(range(offset, offset + len(rewards)))
                
                window = min(20, len(rewards) // 10)
                if window > 1:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    ax.plot(episodes[window-1:], moving_avg, 
                           label=label, color=color, linestyle=linestyle, 
                           linewidth=linewidth, alpha=0.9)
                else:
                    ax.plot(episodes, rewards, label=label, color=color, linestyle=linestyle, 
                           linewidth=linewidth, alpha=0.7)
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Reward', fontweight='bold')
        ax.set_title(f'Окружение {env_id} - Episode Rewards', fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # График длин эпизодов
    for env_idx, env_id in enumerate(env_ids):
        ax = fig.add_subplot(gs[1, env_idx])
        env_data = all_data.get(env_id, {})
        
        # Сначала рисуем все линии с разными стилями
        plots = []
        labels_list = []
        for label, data in env_data.items():
            metrics = data['metrics']
            color = data['color']
            linestyle = data['linestyle']
            linewidth = data['linewidth']
            
            if 'episode_lengths' in metrics and len(metrics['episode_lengths']) > 0:
                lengths = metrics['episode_lengths']
                # Используем правильные индексы эпизодов (начиная с реального первого эпизода)
                if 'episode_indices' in metrics and len(metrics['episode_indices']) == len(lengths):
                    episodes = metrics['episode_indices']
                else:
                    # Если нет сохраненных индексов, создаем от 0
                    offset = metrics.get('episode_offset', 0)
                    episodes = list(range(offset, offset + len(lengths)))
                
                window = min(20, len(lengths) // 10)
                if window > 1:
                    moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
                    plot = ax.plot(episodes[window-1:], moving_avg, 
                           label=label, color=color, linestyle=linestyle, 
                           linewidth=linewidth, alpha=1.0, zorder=10 if label == 'DQN' else 9,
                           marker='o' if label == 'DQN' else 's', markersize=2, markevery=max(1, len(moving_avg)//20))
                else:
                    plot = ax.plot(episodes, lengths, label=label, color=color, linestyle=linestyle, 
                           linewidth=linewidth, alpha=1.0, zorder=10 if label == 'DQN' else 9,
                           marker='o' if label == 'DQN' else 's', markersize=2, markevery=max(1, len(lengths)//20))
                plots.append(plot[0])
                labels_list.append(label)
        
        # Убеждаемся, что обе линии видны - устанавливаем разумные пределы
        if plots:
            y_min = min([p.get_ydata().min() for p in plots if len(p.get_ydata()) > 0])
            y_max = max([p.get_ydata().max() for p in plots if len(p.get_ydata()) > 0])
            if y_max - y_min < 1:  # Если значения почти одинаковые
                y_center = (y_min + y_max) / 2
                ax.set_ylim(y_center - 2, y_center + 2)
            else:
                margin = (y_max - y_min) * 0.1
                ax.set_ylim(max(0, y_min - margin), y_max + margin)
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Length', fontweight='bold')
        ax.set_title(f'Окружение {env_id} - Episode Lengths', fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9, shadow=True, fontsize=9)
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
                step = max(1, len(losses) // 500)
                ax.plot(losses[::step], label=f'{label} Loss', color=color, 
                       linestyle=linestyle, linewidth=linewidth, alpha=0.8)
            elif 'policy_losses' in metrics and len(metrics['policy_losses']) > 0:
                policy_losses = metrics['policy_losses']
                step = max(1, len(policy_losses) // 500)
                ax.plot(policy_losses[::step], label=f'{label} Policy', 
                       color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
        
        ax.set_xlabel('Update Step', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(f'Окружение {env_id} - Training Losses', fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Красивый comparison.png сохранен: {output_path}")
    plt.close()


def main():
    """Создать comparison.png для Task 3."""
    print("="*60)
    print("Генерация comparison.png для Task 3")
    print("="*60)
    
    create_comparison_png(output_path="task3/results/comparison.png")
    
    print("\n" + "="*60)
    print("Готово!")
    print("="*60)
    print(f"\nГрафик сохранен в: task3/results/comparison.png")


if __name__ == "__main__":
    main()

