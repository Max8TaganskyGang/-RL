"""
Proximal Policy Optimization (PPO) алгоритм
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict
from collections import deque

from networks import ActorCritic, ActorCriticLSTM, ActorCriticCNN


class PPOAgent:
    """
    PPO агент с clipping и advantage estimation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: Optional[torch.device] = None,
        use_lstm: bool = False,
        lstm_hidden_dim: int = 128,
        num_layers: int = 1,
        use_cnn: bool = False,
        image_size: int = 28,
    ):
        """
        Args:
            obs_dim: Размерность наблюдения
            action_dim: Количество действий
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: Lambda для GAE (Generalized Advantage Estimation)
            clip_epsilon: Epsilon для clipping в PPO
            value_coef: Коэффициент для value loss
            entropy_coef: Коэффициент для entropy bonus
            max_grad_norm: Максимальная норма градиента для clipping
            device: Устройство для вычислений (CPU/GPU)
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.image_size = image_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        
        # Actor-Critic network
        if use_cnn:
            self.actor_critic = ActorCriticCNN(image_size=image_size, action_dim=action_dim, hidden_dim=128).to(self.device)
        elif use_lstm:
            self.actor_critic = ActorCriticLSTM(obs_dim, action_dim, hidden_dim=128,
                                                lstm_hidden_dim=lstm_hidden_dim, num_layers=num_layers).to(self.device)
            # Для LSTM нужно хранить скрытое состояние
            self.hidden = None
        else:
            self.actor_critic = ActorCritic(obs_dim, action_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
    
    def select_action(self, obs: np.ndarray, reset_hidden: bool = False) -> Tuple[int, float, float]:
        """
        Выбрать действие из policy.
        
        Args:
            obs: Наблюдение (индекс цвета или изображение)
            reset_hidden: Сбросить скрытое состояние LSTM (для нового эпизода)
        
        Returns:
            action: Выбранное действие
            log_prob: Логарифм вероятности действия
            value: Значение состояния
        """
        with torch.no_grad():
            if reset_hidden and self.use_lstm:
                self.hidden = None
            
            # Преобразуем наблюдение в тензор
            if self.use_cnn:
                # obs - изображение [H, W] или [H*W]
                if obs.ndim == 1:
                    # Flatten изображение, нужно reshape
                    obs = obs.reshape(self.image_size, self.image_size)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # [1, H, W]
            else:
                # obs - индекс цвета [1]
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            if self.use_lstm:
                action, log_prob, entropy, value, self.hidden = self.actor_critic.get_action_and_value(
                    obs_tensor, self.hidden
                )
            else:
                action, log_prob, entropy, value = self.actor_critic.get_action_and_value(obs_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычислить Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Награды [T]
            values: Значения состояний [T]
            dones: Флаги завершения [T]
            next_value: Значение следующего состояния
        
        Returns:
            advantages: Преимущества [T]
            returns: Возвраты (target values) [T]
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages[t] = last_gae
            next_value = values[t]
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_value: float = 0.0,
        n_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Обновить policy и value function используя PPO.
        
        Args:
            obs: Наблюдения [T, obs_dim]
            actions: Действия [T]
            old_log_probs: Старые логарифмы вероятностей [T]
            rewards: Награды [T]
            dones: Флаги завершения [T]
            values: Значения состояний [T]
            next_value: Значение следующего состояния
            n_epochs: Количество эпох обучения
            batch_size: Размер батча
        
        Returns:
            Словарь с метриками обучения
        """
        # Вычисляем advantages и returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Нормализация advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Конвертация в тензоры
        if self.use_cnn:
            # Для CNN нужно правильно обработать изображения
            if obs[0].ndim == 1:
                # Flatten изображения
                obs = np.array([o.reshape(self.image_size, self.image_size) for o in obs])
            obs_tensor = torch.FloatTensor(obs).to(self.device)  # [T, H, W]
        else:
            obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Обучение на нескольких эпохах
        indices = np.arange(len(obs))
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(obs), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Получаем новые log_probs и values
                if self.use_lstm:
                    # Для LSTM не используем hidden state в update (каждый батч независим)
                    _, new_log_probs, entropy, new_values, _ = self.actor_critic.get_action_and_value(
                        obs_tensor[batch_indices],
                        None,  # hidden=None для обучения на батчах
                        actions_tensor[batch_indices]
                    )
                elif self.use_cnn:
                    _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(
                        obs_tensor[batch_indices],
                        actions_tensor[batch_indices]
                    )
                else:
                    _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(
                        obs_tensor[batch_indices],
                        actions_tensor[batch_indices]
                    )
                
                # Policy loss (PPO clipping)
                ratio = torch.exp(new_log_probs - old_log_probs_tensor[batch_indices])
                advantages_batch = advantages_tensor[batch_indices]
                
                policy_loss_1 = ratio * advantages_batch
                policy_loss_2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * advantages_batch
                
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(new_values, returns_tensor[batch_indices])
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Оптимизация
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        n_updates = n_epochs * (len(obs) // batch_size + 1)
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }
    
    def save(self, filepath: str):
        """Сохранить модель."""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Загрузить модель."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

