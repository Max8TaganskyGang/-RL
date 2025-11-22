"""
Deep Q-Network (DQN) алгоритм
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from collections import deque

from networks import QNetwork, QNetworkLSTM, QNetworkCNN
from typing import Tuple, Optional
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN агент с experience replay и target network.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
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
            epsilon_start: Начальное значение epsilon для epsilon-greedy
            epsilon_end: Конечное значение epsilon
            epsilon_decay: Скорость затухания epsilon
            buffer_size: Размер replay buffer
            batch_size: Размер батча для обучения
            target_update_freq: Частота обновления target network
            device: Устройство для вычислений (CPU/GPU)
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.image_size = image_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        
        # Q-network и target network
        if use_cnn:
            self.q_network = QNetworkCNN(image_size=image_size, action_dim=action_dim, hidden_dim=128).to(self.device)
            self.target_network = QNetworkCNN(image_size=image_size, action_dim=action_dim, hidden_dim=128).to(self.device)
        elif use_lstm:
            self.q_network = QNetworkLSTM(obs_dim, action_dim, hidden_dim=128, 
                                         lstm_hidden_dim=lstm_hidden_dim, num_layers=num_layers).to(self.device)
            self.target_network = QNetworkLSTM(obs_dim, action_dim, hidden_dim=128,
                                              lstm_hidden_dim=lstm_hidden_dim, num_layers=num_layers).to(self.device)
            # Для LSTM нужно хранить скрытое состояние
            self.hidden = None
            self.target_hidden = None
        else:
            self.q_network = QNetwork(obs_dim, action_dim).to(self.device)
            self.target_network = QNetwork(obs_dim, action_dim).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Счетчик шагов
        self.steps = 0
    
    def select_action(self, obs: np.ndarray, training: bool = True, reset_hidden: bool = False) -> int:
        """
        Выбрать действие используя epsilon-greedy стратегию.
        
        Args:
            obs: Наблюдение (индекс цвета или изображение)
            training: Режим обучения (если False, то epsilon=0)
            reset_hidden: Сбросить скрытое состояние LSTM (для нового эпизода)
        
        Returns:
            Выбранное действие
        """
        if training and np.random.random() < self.epsilon:
            if reset_hidden and self.use_lstm:
                self.hidden = None
            return np.random.randint(self.action_dim)
        
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
                q_values, self.hidden = self.q_network(obs_tensor, self.hidden)
            else:
                q_values = self.q_network(obs_tensor)
            
            action = q_values.argmax().item()
        
        return action
    
    def update_epsilon(self):
        """Обновить epsilon (затухание)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train_step(self) -> Optional[float]:
        """
        Один шаг обучения на батче из replay buffer.
        
        Returns:
            Loss или None, если буфер слишком мал
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Выборка из буфера
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Конвертируем в тензоры
        if self.use_cnn:
            # Для CNN нужно правильно обработать изображения
            if states[0].ndim == 1:
                # Flatten изображения
                states = np.array([s.reshape(self.image_size, self.image_size) for s in states])
                next_states = np.array([s.reshape(self.image_size, self.image_size) for s in next_states])
            states = torch.FloatTensor(states).to(self.device)  # [batch, H, W]
            next_states = torch.FloatTensor(next_states).to(self.device)
        else:
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
        
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Текущие Q-значения
        if self.use_lstm:
            # Для LSTM нужно обрабатывать последовательности
            # В replay buffer храним отдельные наблюдения, поэтому используем hidden=None
            q_values, _ = self.q_network(states, None)
        else:
            q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-значения
        with torch.no_grad():
            if self.use_lstm:
                next_q_values, _ = self.target_network(next_states, None)
            else:
                next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones.float()) * self.gamma * next_q_value
        
        # Loss
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Обновление target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, filepath: str):
        """Сохранить модель."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, filepath)
    
    def load(self, filepath: str):
        """Загрузить модель."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

