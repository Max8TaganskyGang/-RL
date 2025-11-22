"""
Training pipeline для векторизованных сред
Поддержка как SyncVectorEnv, так и полноценной векторизации
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from collections import deque
import time
import os
import sys

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from dqn import DQNAgent
from ppo import PPOAgent


def train_dqn_vectorized(
    env,
    agent: DQNAgent,
    n_episodes: int = 1000,
    max_steps_per_episode: int = 200,
    train_freq: int = 4,
    eval_freq: int = 50,
    eval_episodes: int = 10,
    save_dir: Optional[str] = None,
    use_wandb: bool = False,
    wandb_config: Optional[Dict] = None,
) -> Dict[str, List[float]]:
    """
    Обучение DQN агента на векторизованной среде.
    
    Args:
        env: Векторизованная среда (SyncVectorEnv или VectorizedGridWorldEnv)
        agent: DQN агент
        n_episodes: Количество эпизодов обучения (суммарно по всем средам)
        max_steps_per_episode: Максимальное количество шагов в эпизоде
        train_freq: Частота обучения (каждые N шагов)
        eval_freq: Частота оценки (каждые N эпизодов)
        eval_episodes: Количество эпизодов для оценки
        save_dir: Директория для сохранения результатов
        use_wandb: Использовать ли Wandb для логирования
        wandb_config: Конфигурация для Wandb
    
    Returns:
        Словарь с метриками обучения
    """
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
        'eval_rewards': [],
        'eval_lengths': [],
        'total_steps': [],
    }
    
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    # Определяем количество параллельных сред
    if hasattr(env, 'num_envs'):
        num_envs = env.num_envs
    elif hasattr(env, 'num_envs'):
        num_envs = env.num_envs
    else:
        num_envs = getattr(env, 'num_envs', 1)
    
    # Инициализация wandb
    if use_wandb and WANDB_AVAILABLE:
        if wandb_config is None:
            wandb_config = {}
        try:
            wandb.init(
                entity='sinitskii-mi',
                project=wandb_config.get('project', 'gridworld-vectorized'),
                name=wandb_config.get('name', 'dqn-vectorized'),
                config={
                    **wandb_config.get('config', {}),
                    'num_envs': num_envs,
                },
                reinit=True,
                mode='online'
            )
        except Exception as e:
            print(f"⚠️  Ошибка Wandb: {e}. Продолжаю обучение без логирования в Wandb.")
            use_wandb = False
    
    # Сброс среды
    obs, info = env.reset()
    # obs shape: (num_envs, obs_shape)
    
    total_steps = 0
    episode_count = 0
    step_count = 0
    
    # Буферы для накопления данных
    episode_rewards_buf = np.zeros(num_envs)
    episode_lengths_buf = np.zeros(num_envs)
    done_buf = np.zeros(num_envs, dtype=bool)
    
    start_time = time.time()
    
    print(f"🚀 Начало обучения DQN на векторизованной среде ({num_envs} параллельных сред)")
    print(f"📊 Общее количество эпизодов: {n_episodes}")
    print()
    
    while episode_count < n_episodes:
        # Выбор действий для всех сред
        if hasattr(agent, 'use_lstm') and agent.use_lstm:
            # Для LSTM нужно обрабатывать каждую среду отдельно
            actions = []
            for i in range(num_envs):
                if done_buf[i]:
                    # Сброс скрытого состояния для завершенных сред
                    agent.reset_hidden_state()
                action = agent.select_action(obs[i], training=True, reset_hidden=done_buf[i])
                actions.append(action)
            actions = np.array(actions)
        else:
            # Для обычного DQN можем обработать batch
            actions = []
            for i in range(num_envs):
                if not done_buf[i]:  # Только для активных сред
                    action = agent.select_action(obs[i], training=True)
                    actions.append(action)
                else:
                    actions.append(0)  # Dummy action для завершенных сред
            actions = np.array(actions)
        
        # Выполнение действий во всех средах
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated | truncated
        
        # Обновляем буферы для активных сред
        active_mask = ~done_buf
        episode_rewards_buf[active_mask] += rewards[active_mask]
        episode_lengths_buf[active_mask] += 1
        
        # Обработка завершенных эпизодов
        newly_done = done & ~done_buf
        if np.any(newly_done):
            for i in np.where(newly_done)[0]:
                episode_rewards.append(float(episode_rewards_buf[i]))
                episode_lengths.append(int(episode_lengths_buf[i]))
                
                metrics['episode_rewards'].append(float(episode_rewards_buf[i]))
                metrics['episode_lengths'].append(int(episode_lengths_buf[i]))
                
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'episode_reward': float(episode_rewards_buf[i]),
                        'episode_length': int(episode_lengths_buf[i]),
                        'episode': episode_count,
                        'total_steps': total_steps,
                    })
                
                episode_count += 1
                
                # Сброс буферов для завершенных сред
                episode_rewards_buf[i] = 0
                episode_lengths_buf[i] = 0
                done_buf[i] = False  # Среда будет автоматически сброшена
                
                if episode_count % 100 == 0:
                    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                    avg_length = np.mean(episode_lengths) if episode_lengths else 0
                    elapsed = time.time() - start_time
                    print(f"Episode {episode_count}/{n_episodes} | "
                          f"Avg Reward: {avg_reward:.2f} | "
                          f"Avg Length: {avg_length:.2f} | "
                          f"Total Steps: {total_steps} | "
                          f"Time: {elapsed:.1f}s")
        
        # Обучение агента
        step_count += 1
        total_steps += num_envs
        
        if step_count % train_freq == 0 and len(episode_rewards) > 0:
            # Обучаем агента на данных из активных сред
            # Здесь нужно накапливать transitions для обучения
            # Для упрощения, мы будем обучать на каждом шаге из replay buffer
            # Но это требует доработки replay buffer для работы с batch
            pass
        
        # Сохранение переходов в replay buffer (для каждого активного окружения)
        for i in range(num_envs):
            if not done_buf[i]:
                agent.replay_buffer.push(
                    obs[i],
                    actions[i],
                    rewards[i],
                    next_obs[i],
                    done[i]
                )
                
                # Обучение на batch из replay buffer
                if len(agent.replay_buffer) > agent.batch_size and step_count % train_freq == 0:
                    loss = agent.train_step()
                    if loss is not None:
                        metrics['losses'].append(loss)
                        if use_wandb and WANDB_AVAILABLE:
                            wandb.log({
                                'loss': loss,
                                'total_steps': total_steps,
                            })
        
        # Обновление наблюдений
        obs = next_obs
        done_buf = done
        
        # Сброс завершенных сред
        if np.any(done):
            reset_indices = np.where(done)[0]
            for i in reset_indices:
                # Сбрасываем только завершенные среды
                # Для SyncVectorEnv это делается автоматически
                # Для VectorizedGridWorldEnv нужно вызывать reset только для нужных сред
                if hasattr(env, 'reset_single'):
                    obs[i], _ = env.reset_single(i)
                else:
                    # Если среда не поддерживает частичный reset, сбрасываем все
                    obs, info = env.reset()
                    break
        
        # Оценка
        if episode_count > 0 and episode_count % eval_freq == 0:
            eval_reward, eval_length = evaluate_vectorized(env, agent, eval_episodes)
            metrics['eval_rewards'].append(eval_reward)
            metrics['eval_lengths'].append(eval_length)
            
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'eval_reward': eval_reward,
                    'eval_length': eval_length,
                    'eval_episode': episode_count,
                })
            
            print(f"Evaluation | Avg Reward: {eval_reward:.2f} | Avg Length: {eval_length:.2f}")
            
            # Сохранение модели
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(agent.q_network.state_dict(), 
                          os.path.join(save_dir, f'dqn_episode_{episode_count}.pt'))
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Обучение завершено!")
    print(f"📊 Всего эпизодов: {episode_count}")
    print(f"📊 Всего шагов: {total_steps}")
    print(f"⏱️  Время обучения: {elapsed_time:.2f}s")
    print(f"⚡ Шагов в секунду: {total_steps / elapsed_time:.2f}")
    
    # Сохранение метрик
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import pickle
        with open(os.path.join(save_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return metrics


def evaluate_vectorized(env, agent, n_episodes: int = 10) -> tuple:
    """Оценка агента на векторизованной среде."""
    episode_rewards = []
    episode_lengths = []
    
    obs, info = env.reset()
    done_buf = np.zeros(env.num_envs, dtype=bool)
    episode_rewards_buf = np.zeros(env.num_envs)
    episode_lengths_buf = np.zeros(env.num_envs)
    
    episode_count = 0
    
    while episode_count < n_episodes:
        # Выбор действий
        actions = []
        for i in range(env.num_envs):
            if not done_buf[i]:
                if hasattr(agent, 'use_lstm') and agent.use_lstm:
                    action = agent.select_action(obs[i], training=False, reset_hidden=done_buf[i])
                else:
                    action = agent.select_action(obs[i], training=False)
                actions.append(action)
            else:
                actions.append(0)
        actions = np.array(actions)
        
        # Выполнение действий
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated | truncated
        
        # Обновление буферов
        active_mask = ~done_buf
        episode_rewards_buf[active_mask] += rewards[active_mask]
        episode_lengths_buf[active_mask] += 1
        
        # Обработка завершенных эпизодов
        newly_done = done & ~done_buf
        if np.any(newly_done):
            for i in np.where(newly_done)[0]:
                episode_rewards.append(float(episode_rewards_buf[i]))
                episode_lengths.append(int(episode_lengths_buf[i]))
                episode_rewards_buf[i] = 0
                episode_lengths_buf[i] = 0
                done_buf[i] = False
                episode_count += 1
        
        obs = next_obs
        done_buf = done
        
        # Сброс завершенных сред
        if np.any(done):
            obs, info = env.reset()
    
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    
    return avg_reward, avg_length

