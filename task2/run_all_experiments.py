"""
Запуск всех экспериментов по одному разу
"""

import subprocess
import sys
import os


def run_experiment(method, use_lstm, env_id, episodes=500):
    """Запустить один эксперимент."""
    lstm_flag = "--use-lstm" if use_lstm else ""
    cmd = [
        sys.executable, "task2/train_task1.py",
        "--method", method,
        "--env", str(env_id),
        "--episodes", str(episodes),
        "--wandb",
        "--wandb-project", "RL2"
    ]
    if lstm_flag:
        cmd.append(lstm_flag)
    
    print(f"\n{'='*60}")
    print(f"Запуск: {method} {'LSTM' if use_lstm else 'no-LSTM'} env{env_id}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✓ Успешно: {method} {'LSTM' if use_lstm else 'no-LSTM'} env{env_id}")
    else:
        print(f"✗ Ошибка: {method} {'LSTM' if use_lstm else 'no-LSTM'} env{env_id}")
    
    return result.returncode == 0


def main():
    """Запустить все эксперименты по одному разу."""
    experiments = [
        ("dqn", False, 1), ("dqn", False, 2), ("dqn", False, 3),
        ("dqn", True, 1), ("dqn", True, 2), ("dqn", True, 3),
        ("ppo", False, 1), ("ppo", False, 2), ("ppo", False, 3),
        ("ppo", True, 1), ("ppo", True, 2), ("ppo", True, 3),
    ]
    
    print("="*60)
    print("Запуск всех экспериментов (по одному разу)")
    print("="*60)
    print(f"Всего экспериментов: {len(experiments)}")
    
    results = []
    for method, use_lstm, env_id in experiments:
        success = run_experiment(method, use_lstm, env_id, episodes=500)
        results.append((method, use_lstm, env_id, success))
    
    print("\n" + "="*60)
    print("Результаты:")
    print("="*60)
    for method, use_lstm, env_id, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {method} {'LSTM' if use_lstm else 'no-LSTM'} env{env_id}")
    
    successful = sum(1 for _, _, _, s in results if s)
    print(f"\nУспешно: {successful}/{len(results)}")


if __name__ == "__main__":
    main()
