"""
Replay Buffer для DQN
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional


class ReplayBuffer:
    """
    Буфер воспроизведения опыта для DQN.
    Хранит переходы (state, action, reward, next_state, done).
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Максимальный размер буфера
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Добавить переход в буфер.
        
        Args:
            state: Текущее состояние
            action: Выполненное действие
            reward: Полученная награда
            next_state: Следующее состояние
            done: Флаг завершения эпизода
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                np.ndarray, np.ndarray]:
        """
        Выбрать случайную выборку из буфера.
        
        Args:
            batch_size: Размер выборки
        
        Returns:
            states, actions, rewards, next_states, dones
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = np.array([self.buffer[i][0] for i in indices])
        actions = np.array([self.buffer[i][1] for i in indices])
        rewards = np.array([self.buffer[i][2] for i in indices])
        next_states = np.array([self.buffer[i][3] for i in indices])
        dones = np.array([self.buffer[i][4] for i in indices])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Текущий размер буфера."""
        return len(self.buffer)

