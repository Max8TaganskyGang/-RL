"""
Нейронные сети для DQN и PPO
Поддержка как простых наблюдений (координаты), так и изображений (MNIST)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class QNetwork(nn.Module):
    """
    Q-Network для DQN.
    Принимает наблюдение (позицию агента) и возвращает Q-значения для каждого действия.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        """
        Args:
            obs_dim: Размерность наблюдения (2 для позиции)
            action_dim: Количество действий (4)
            hidden_dims: Размеры скрытых слоев
        """
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = obs_dim
        
        # Создаем скрытые слои
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Выходной слой
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Инициализация весов
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Инициализация весов."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Наблюдение [batch_size, obs_dim]
        
        Returns:
            Q-значения для каждого действия [batch_size, action_dim]
        """
        return self.network(obs)


class ActorCritic(nn.Module):
    """
    Actor-Critic сеть для PPO.
    Actor (policy) и Critic (value function) с общим энкодером.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        """
        Args:
            obs_dim: Размерность наблюдения (2 для позиции)
            action_dim: Количество действий (4)
            hidden_dims: Размеры скрытых слоев
        """
        super(ActorCritic, self).__init__()
        
        # Общий энкодер
        encoder_layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Actor head (policy)
        self.actor = nn.Linear(input_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(input_dim, 1)
        
        # Инициализация весов
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Инициализация весов."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: Наблюдение [batch_size, obs_dim]
        
        Returns:
            action_logits: Логиты действий [batch_size, action_dim]
            value: Значение состояния [batch_size, 1]
        """
        features = self.encoder(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """
        Получить действие и значение для PPO.
        
        Args:
            obs: Наблюдение [batch_size, obs_dim]
            action: Действие (если None, то выбирается из policy)
        
        Returns:
            action: Выбранное действие
            log_prob: Логарифм вероятности действия
            entropy: Энтропия распределения
            value: Значение состояния
        """
        action_logits, value = self.forward(obs)
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


class CNNEncoder(nn.Module):
    """
    CNN энкодер для обработки изображений (MNIST).
    Преобразует изображение в вектор признаков.
    """
    
    def __init__(self, input_size: int = 14, output_dim: int = 128):
        """
        Args:
            input_size: Размер входного изображения (14x14 или 28x28)
            output_dim: Размерность выходного вектора признаков
        """
        super(CNNEncoder, self).__init__()
        
        # CNN слои для извлечения признаков из изображения
        self.conv_layers = nn.Sequential(
            # Первый сверточный блок
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7 или 28x28 -> 14x14
            
            # Второй сверточный блок
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7 -> 3x3 или 14x14 -> 7x7
            
            # Третий сверточный блок (опционально, для больших изображений)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Вычисляем размер после сверточных слоев динамически
        # После двух MaxPool2d: input_size -> input_size/4
        # После трех сверточных блоков размер не меняется
        conv_output_size = (input_size // 4) * (input_size // 4) * 64
        
        # Полносвязные слои
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Изображение [batch_size, 1, H, W] или [batch_size, H, W]
        
        Returns:
            Вектор признаков [batch_size, output_dim]
        """
        # Добавляем channel dimension если нужно
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, H, W] -> [batch, 1, H, W]
        
        # Применяем CNN
        x = self.conv_layers(x)
        
        # Применяем полносвязные слои
        x = self.fc_layers(x)
        
        return x


class QNetworkCNN(nn.Module):
    """
    Q-Network с CNN энкодером для изображений.
    """
    
    def __init__(self, image_size: int = 14, action_dim: int = 4, hidden_dim: int = 128):
        """
        Args:
            image_size: Размер входного изображения
            action_dim: Количество действий
            hidden_dim: Размерность скрытого слоя
        """
        super(QNetworkCNN, self).__init__()
        
        # CNN энкодер
        self.encoder = CNNEncoder(input_size=image_size, output_dim=hidden_dim)
        
        # Q-value head
        self.q_head = nn.Linear(hidden_dim, action_dim)
        
        # Инициализация
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Наблюдение [batch_size, H, W] или [batch_size, 1, H, W]
        
        Returns:
            Q-значения [batch_size, action_dim]
        """
        features = self.encoder(obs)
        q_values = self.q_head(features)
        return q_values


class ActorCriticCNN(nn.Module):
    """
    Actor-Critic сеть с CNN энкодером для изображений.
    """
    
    def __init__(self, image_size: int = 14, action_dim: int = 4, hidden_dim: int = 128):
        """
        Args:
            image_size: Размер входного изображения
            action_dim: Количество действий
            hidden_dim: Размерность скрытого слоя
        """
        super(ActorCriticCNN, self).__init__()
        
        # Общий CNN энкодер
        self.encoder = CNNEncoder(input_size=image_size, output_dim=hidden_dim)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Инициализация
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: Наблюдение [batch_size, H, W] или [batch_size, 1, H, W]
        
        Returns:
            action_logits: Логиты действий [batch_size, action_dim]
            value: Значение состояния [batch_size, 1]
        """
        features = self.encoder(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """
        Получить действие и значение для PPO.
        
        Args:
            obs: Наблюдение [batch_size, H, W] или [batch_size, 1, H, W]
            action: Действие (если None, то выбирается из policy)
        
        Returns:
            action: Выбранное действие
            log_prob: Логарифм вероятности действия
            entropy: Энтропия распределения
            value: Значение состояния
        """
        action_logits, value = self.forward(obs)
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)

class QNetworkLSTM(nn.Module):
    """
    Q-Network с LSTM для обработки последовательностей наблюдений.
    Используется для POMDP задач, где нужна память.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, lstm_hidden_dim: int = 128, num_layers: int = 1):
        """
        Args:
            obs_dim: Размерность одного наблюдения (1 для индекса цвета)
            action_dim: Количество действий (4)
            hidden_dim: Размерность скрытого слоя после LSTM
            lstm_hidden_dim: Размерность скрытого состояния LSTM
            num_layers: Количество слоев LSTM
        """
        super(QNetworkLSTM, self).__init__()
        
        # Embedding для наблюдения (если obs_dim = 1, то это индекс цвета)
        self.obs_embedding = nn.Embedding(100, 32)  # Поддерживаем до 100 цветов
        
        # LSTM для обработки последовательности
        self.lstm = nn.LSTM(
            input_size=32,  # Размер после embedding
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Полносвязные слои после LSTM
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Инициализация
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Инициализация весов."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Args:
            obs: Наблюдения [batch_size, seq_len, obs_dim] или [batch_size, obs_dim] (seq_len=1)
            hidden: Скрытое состояние LSTM (h_n, c_n)
        
        Returns:
            Q-значения [batch_size, action_dim]
            hidden: Новое скрытое состояние
        """
        # Если obs имеет 2 измерения, добавляем seq_len=1
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # [batch, obs_dim] -> [batch, 1, obs_dim]
        
        batch_size, seq_len, _ = obs.shape
        
        # Преобразуем наблюдения в embedding
        # obs должен быть int для embedding
        obs_int = obs.long().squeeze(-1)  # [batch, seq_len]
        obs_embedded = self.obs_embedding(obs_int)  # [batch, seq_len, embedding_dim]
        
        # Применяем LSTM
        lstm_out, hidden = self.lstm(obs_embedded, hidden)  # [batch, seq_len, lstm_hidden]
        
        # Берем последний выход LSTM
        if seq_len == 1:
            lstm_out = lstm_out.squeeze(1)  # [batch, lstm_hidden]
        else:
            lstm_out = lstm_out[:, -1, :]  # [batch, lstm_hidden]
        
        # Применяем полносвязные слои
        q_values = self.fc(lstm_out)  # [batch, action_dim]
        
        return q_values, hidden


class ActorCriticLSTM(nn.Module):
    """
    Actor-Critic сеть с LSTM для обработки последовательностей наблюдений.
    Используется для POMDP задач, где нужна память.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, lstm_hidden_dim: int = 128, num_layers: int = 1):
        """
        Args:
            obs_dim: Размерность одного наблюдения (1 для индекса цвета)
            action_dim: Количество действий (4)
            hidden_dim: Размерность скрытого слоя после LSTM
            lstm_hidden_dim: Размерность скрытого состояния LSTM
            num_layers: Количество слоев LSTM
        """
        super(ActorCriticLSTM, self).__init__()
        
        # Embedding для наблюдения
        self.obs_embedding = nn.Embedding(100, 32)  # Поддерживаем до 100 цветов
        
        # LSTM для обработки последовательности
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Общий энкодер после LSTM
        self.encoder = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Инициализация
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Инициализация весов."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Args:
            obs: Наблюдения [batch_size, seq_len, obs_dim] или [batch_size, obs_dim]
            hidden: Скрытое состояние LSTM
        
        Returns:
            action_logits: Логиты действий [batch_size, action_dim]
            value: Значение состояния [batch_size, 1]
            hidden: Новое скрытое состояние
        """
        # Если obs имеет 2 измерения, добавляем seq_len=1
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # [batch, obs_dim] -> [batch, 1, obs_dim]
        
        batch_size, seq_len, _ = obs.shape
        
        # Преобразуем наблюдения в embedding
        obs_int = obs.long().squeeze(-1)  # [batch, seq_len]
        obs_embedded = self.obs_embedding(obs_int)  # [batch, seq_len, embedding_dim]
        
        # Применяем LSTM
        lstm_out, hidden = self.lstm(obs_embedded, hidden)  # [batch, seq_len, lstm_hidden]
        
        # Берем последний выход LSTM
        if seq_len == 1:
            lstm_out = lstm_out.squeeze(1)  # [batch, lstm_hidden]
        else:
            lstm_out = lstm_out[:, -1, :]  # [batch, lstm_hidden]
        
        # Применяем энкодер
        features = self.encoder(lstm_out)  # [batch, hidden_dim]
        
        # Actor и Critic
        action_logits = self.actor(features)  # [batch, action_dim]
        value = self.critic(features)  # [batch, 1]
        
        return action_logits, value, hidden
    
    def get_action_and_value(self, obs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, action: torch.Tensor = None):
        """
        Получить действие и значение для PPO.
        
        Args:
            obs: Наблюдение [batch_size, obs_dim] или [batch_size, seq_len, obs_dim]
            hidden: Скрытое состояние LSTM
            action: Действие (если None, то выбирается из policy)
        
        Returns:
            action: Выбранное действие
            log_prob: Логарифм вероятности действия
            entropy: Энтропия распределения
            value: Значение состояния
            hidden: Новое скрытое состояние
        """
        action_logits, value, hidden = self.forward(obs, hidden)
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1), hidden

