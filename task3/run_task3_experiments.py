"""
Запуск всех экспериментов для Task 3 (MNIST)
"""

import subprocess
import sys
import os


def run_experiment(method, env_id, episodes=100):
    """Запустить один эксперимент."""
    cmd = [
        sys.executable, "task3/train_task3.py",
        "--method", method,
        "--env", str(env_id),
        "--episodes", str(episodes),
        "--wandb",
        "--wandb-project", "RL3"
    ]
    
    print(f"\n{'='*60}")
    print(f"Запуск: {method.upper()} env{env_id}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✓ Успешно: {method} env{env_id}")
    else:
        print(f"✗ Ошибка: {method} env{env_id}")
    
    return result.returncode == 0


def main():
    """Запустить все эксперименты для Task 3."""
    experiments = [
        # DQN
        ("dqn", 1),
        ("dqn", 2),
        ("dqn", 3),
        # PPO
        ("ppo", 1),
        ("ppo", 2),
        ("ppo", 3),
    ]
    
    print("="*60)
    print("Запуск всех экспериментов Task 3 (MNIST)")
    print("="*60)
    print(f"Всего экспериментов: {len(experiments)}")
    print(f"Проект Wandb: RL3")
    
    results = []
    for method, env_id in experiments:
        success = run_experiment(method, env_id, episodes=100)
        results.append((method, env_id, success))
    
    print("\n" + "="*60)
    print("Результаты:")
    print("="*60)
    for method, env_id, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {method} env{env_id}")
    
    successful = sum(1 for _, _, s in results if s)
    print(f"\nУспешно: {successful}/{len(results)}")


if __name__ == "__main__":
    main()

