"""
Pipeline обучения для DQN и PPO
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from collections import deque
import matplotlib.pyplot as plt
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from dqn import DQNAgent
from ppo import PPOAgent
from gridworld_env import GridWorldEnv


def train_dqn(
    env: GridWorldEnv,
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
    Обучение DQN агента.
    
    Args:
        env: Окружение
        agent: DQN агент
        n_episodes: Количество эпизодов обучения
        max_steps_per_episode: Максимальное количество шагов в эпизоде
        train_freq: Частота обучения (каждые N шагов)
        eval_freq: Частота оценки (каждые N эпизодов)
        eval_episodes: Количество эпизодов для оценки
        save_dir: Директория для сохранения результатов
    
    Returns:
        Словарь с метриками обучения
    """
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
        'eval_rewards': [],
        'eval_lengths': [],
    }
    
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    # Инициализация wandb
    if use_wandb and WANDB_AVAILABLE:
        if wandb_config is None:
            wandb_config = {}
        try:
            wandb.init(
                entity='sinitskii-mi',
                project=wandb_config.get('project', 'gridworld-dqn'),
                name=wandb_config.get('name', 'dqn-experiment'),
                config=wandb_config.get('config', {}),
                reinit=True,
                mode='online'
            )
        except Exception as e:
            print(f"⚠️  Ошибка Wandb: {e}. Продолжаю обучение без логирования в Wandb.")
            use_wandb = False
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        # Сброс скрытого состояния LSTM для нового эпизода
        reset_hidden = True
        
        for step in range(max_steps_per_episode):
            # Выбор действия
            if hasattr(agent, 'use_lstm') and agent.use_lstm:
                action = agent.select_action(obs, training=True, reset_hidden=reset_hidden)
                reset_hidden = False  # Только первый шаг эпизода
            else:
                action = agent.select_action(obs, training=True)
            
            # Выполнение действия
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Сохранение в replay buffer
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            # Обучение
            if len(agent.replay_buffer) >= agent.batch_size and step % train_freq == 0:
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # Обновление epsilon
        agent.update_epsilon()
        
        # Сохранение метрик
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            metrics['losses'].append(avg_loss)
        
        # Логирование в wandb
        if use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'episode': episode + 1,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'epsilon': agent.epsilon,
            }
            if episode_losses:
                log_dict['loss'] = np.mean(episode_losses)
            if len(episode_rewards) > 0:
                log_dict['avg_reward_100'] = np.mean(episode_rewards)
                log_dict['avg_length_100'] = np.mean(episode_lengths)
            wandb.log(log_dict)
        
        # Логирование
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Оценка
        if (episode + 1) % eval_freq == 0:
            eval_reward, eval_length = evaluate_dqn(env, agent, eval_episodes)
            metrics['eval_rewards'].append(eval_reward)
            metrics['eval_lengths'].append(eval_length)
            print(f"Evaluation | Avg Reward: {eval_reward:.2f} | Avg Length: {eval_length:.2f}")
            
            # Логирование оценки в wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'eval_reward': eval_reward,
                    'eval_length': eval_length,
                    'eval_episode': episode + 1,
                })
            
            # Сохранение модели
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                agent.save(os.path.join(save_dir, f'dqn_episode_{episode + 1}.pt'))
    
    # Завершение wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    # Сохранение метрик
    if save_dir:
        import pickle
        os.makedirs(save_dir, exist_ok=True)
        metrics_file = os.path.join(save_dir, 'metrics.pkl')
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)
    
    return metrics


def train_ppo(
    env: GridWorldEnv,
    agent: PPOAgent,
    n_episodes: int = 1000,
    max_steps_per_episode: int = 200,
    update_freq: int = 512,  # Количество шагов перед обновлением
    n_epochs: int = 10,
    batch_size: int = 64,
    eval_freq: int = 50,
    eval_episodes: int = 10,
    save_dir: Optional[str] = None,
    use_wandb: bool = False,
    wandb_config: Optional[Dict] = None,
) -> Dict[str, List[float]]:
    """
    Обучение PPO агента.
    
    Args:
        env: Окружение
        agent: PPO агент
        n_episodes: Количество эпизодов обучения
        max_steps_per_episode: Максимальное количество шагов в эпизоде
        update_freq: Частота обновления (каждые N шагов)
        n_epochs: Количество эпох обучения на одном батче
        batch_size: Размер батча для обучения
        eval_freq: Частота оценки (каждые N эпизодов)
        eval_episodes: Количество эпизодов для оценки
        save_dir: Директория для сохранения результатов
    
    Returns:
        Словарь с метриками обучения
    """
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'eval_rewards': [],
        'eval_lengths': [],
    }
    
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    # Инициализация wandb
    if use_wandb and WANDB_AVAILABLE:
        if wandb_config is None:
            wandb_config = {}
        try:
            wandb.init(
                entity='sinitskii-mi',
                project=wandb_config.get('project', 'gridworld-ppo'),
                name=wandb_config.get('name', 'ppo-experiment'),
                config=wandb_config.get('config', {}),
                reinit=True,
                mode='online'
            )
        except Exception as e:
            print(f"⚠️  Ошибка Wandb: {e}. Продолжаю обучение без логирования в Wandb.")
            use_wandb = False
    
    # Буферы для сбора данных
    obs_buffer = []
    action_buffer = []
    log_prob_buffer = []
    reward_buffer = []
    done_buffer = []
    value_buffer = []
    
    episode = 0
    total_steps = 0
    
    while episode < n_episodes:
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Сброс скрытого состояния LSTM для нового эпизода
        reset_hidden = True
        
        for step in range(max_steps_per_episode):
            # Выбор действия
            if hasattr(agent, 'use_lstm') and agent.use_lstm:
                action, log_prob, value = agent.select_action(obs, reset_hidden=reset_hidden)
                reset_hidden = False  # Только первый шаг эпизода
            else:
                action, log_prob, value = agent.select_action(obs)
            
            # Выполнение действия
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Сохранение данных
            obs_buffer.append(obs)
            action_buffer.append(action)
            log_prob_buffer.append(log_prob)
            reward_buffer.append(reward)
            done_buffer.append(done)
            value_buffer.append(value)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            if done:
                episode += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                metrics['episode_rewards'].append(episode_reward)
                metrics['episode_lengths'].append(episode_length)
                
                # Логирование в wandb
                if use_wandb and WANDB_AVAILABLE:
                    log_dict = {
                        'episode': episode,
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                    }
                    if len(episode_rewards) > 0:
                        log_dict['avg_reward_100'] = np.mean(episode_rewards)
                        log_dict['avg_length_100'] = np.mean(episode_lengths)
                    wandb.log(log_dict)
                
                # Логирование
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards)
                    avg_length = np.mean(episode_lengths)
                    print(f"Episode {episode}/{n_episodes} | "
                          f"Avg Reward: {avg_reward:.2f} | "
                          f"Avg Length: {avg_length:.2f}")
                
                # Оценка
                if episode % eval_freq == 0:
                    eval_reward, eval_length = evaluate_ppo(env, agent, eval_episodes)
                    metrics['eval_rewards'].append(eval_reward)
                    metrics['eval_lengths'].append(eval_length)
                    print(f"Evaluation | Avg Reward: {eval_reward:.2f} | Avg Length: {eval_length:.2f}")
                    
                    # Логирование оценки в wandb
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            'eval_reward': eval_reward,
                            'eval_length': eval_length,
                            'eval_episode': episode,
                        })
                    
                    # Сохранение модели
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        agent.save(os.path.join(save_dir, f'ppo_episode_{episode}.pt'))
                
                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Обновление policy
            if total_steps % update_freq == 0 and len(obs_buffer) > 0:
                # Получаем значение следующего состояния (если эпизод не завершен)
                if not done:
                    _, _, next_value = agent.select_action(obs)
                else:
                    next_value = 0.0
                
                # Обновление
                update_metrics = agent.update(
                    np.array(obs_buffer),
                    np.array(action_buffer),
                    np.array(log_prob_buffer),
                    np.array(reward_buffer),
                    np.array(done_buffer),
                    np.array(value_buffer),
                    next_value,
                    n_epochs=n_epochs,
                    batch_size=batch_size
                )
                
                metrics['policy_losses'].extend([update_metrics['policy_loss']] * len(obs_buffer))
                metrics['value_losses'].extend([update_metrics['value_loss']] * len(obs_buffer))
                metrics['entropies'].extend([update_metrics['entropy']] * len(obs_buffer))
                
                # Логирование метрик обновления в wandb
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'policy_loss': update_metrics['policy_loss'],
                        'value_loss': update_metrics['value_loss'],
                        'entropy': update_metrics['entropy'],
                        'total_loss': update_metrics['policy_loss'] + update_metrics['value_loss'] * 0.5 - update_metrics['entropy'] * 0.01,
                        'update_step': total_steps // update_freq,
                        'total_steps': total_steps,
                    })
                
                # Очистка буферов
                obs_buffer = []
                action_buffer = []
                log_prob_buffer = []
                reward_buffer = []
                done_buffer = []
                value_buffer = []
    
    # Завершение wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    # Сохранение метрик
    if save_dir:
        import pickle
        os.makedirs(save_dir, exist_ok=True)
        metrics_file = os.path.join(save_dir, 'metrics.pkl')
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)
    
    return metrics


def evaluate_dqn(env: GridWorldEnv, agent: DQNAgent, n_episodes: int = 10) -> tuple:
    """Оценка DQN агента."""
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return np.mean(episode_rewards), np.mean(episode_lengths)


def evaluate_ppo(env: GridWorldEnv, agent: PPOAgent, n_episodes: int = 10) -> tuple:
    """Оценка PPO агента."""
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _, _ = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return np.mean(episode_rewards), np.mean(episode_lengths)

