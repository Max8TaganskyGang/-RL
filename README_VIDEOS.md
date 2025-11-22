# Видео и графики результатов

## Видео

После завершения обучения все видео будут сохранены в папке:
```
task2_videos/
```

Видео будут называться:
- `dqn-no-lstm-env1.mp4`, `dqn-no-lstm-env2.mp4`, `dqn-no-lstm-env3.mp4`
- `dqn-lstm-env1.mp4`, `dqn-lstm-env2.mp4`, `dqn-lstm-env3.mp4`
- `ppo-no-lstm-env1.mp4`, `ppo-no-lstm-env2.mp4`, `ppo-no-lstm-env3.mp4`
- `ppo-lstm-env1.mp4`, `ppo-lstm-env2.mp4`, `ppo-lstm-env3.mp4`

**Для создания видео запустите:**
```bash
python record_all_videos.py
```

## Графики сравнения

Графики сравнения всех методов будут сохранены в папке:
```
results/comparisons/
```

**Для создания графиков запустите:**
```bash
python compare_all_methods.py
```

Будут созданы:
- `comparison_env1.png` - сравнение всех методов на окружении 1
- `comparison_env2.png` - сравнение всех методов на окружении 2
- `comparison_env3.png` - сравнение всех методов на окружении 3
- `summary_comparison.png` - общий график сравнения на всех окружениях

## Wandb

Все метрики логируются в Wandb проект:
https://wandb.ai/maxsir195-yandex/RL2

