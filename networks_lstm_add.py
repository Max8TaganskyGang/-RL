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

