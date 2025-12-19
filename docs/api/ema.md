# EMA (Exponential Moving Average)

**Модуль:** `transformers.ema`

EMA поддерживает сглаженную версию весов модели, которая обычно показывает лучшую generalization чем финальные веса обучения.

---

## 📊 Зачем нужен EMA?

| Проблема | Решение EMA |
|----------|-------------|
| Веса "прыгают" после каждого batch | EMA сглаживает изменения |
| Шум в градиентах | Усреднение убирает шум |
| Переобучение | EMA веса более стабильны |

**Типичное улучшение:** +1-3% на eval метриках*

> ⚠️ **Важно (v1.1.0):** Улучшение +1-3% достигается на **моделях >1B параметров** при длительном обучении (10k+ steps). На маленьких моделях эффект может быть минимальным. См. [RESEARCH.md](/docs/RESEARCH.md) для деталей.

---

## 🔧 Быстрый старт

```python
from transformers import Trainer
from transformers.ema import EMACallback

# Создаём callback
ema_callback = EMACallback(decay=0.999)

# Добавляем в Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[ema_callback]
)

# Обучаем
trainer.train()

# Применяем EMA веса (более качественные!)
ema_callback.apply_ema(model)
```

---

## 📖 API Reference

### EMACallback

```python
class EMACallback(TrainerCallback):
    def __init__(
        self,
        decay: float = 0.999,
        update_after_step: int = 0,
        update_every: int = 1
    )
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `decay` | float | 0.999 | Коэффициент сглаживания. Выше = медленнее обновления |
| `update_after_step` | int | 0 | Начать EMA после N шагов (warmup) |
| `update_every` | int | 1 | Обновлять каждые N шагов |

**Методы:**

| Метод | Описание |
|-------|----------|
| `apply_ema(model)` | Применить EMA веса к модели |
| `restore_original(model)` | Восстановить оригинальные веса |
| `get_ema_state()` | Получить EMA state для сохранения |
| `load_ema_state(state_dict)` | Загрузить EMA state |

---

### EMAModel

```python
class EMAModel:
    def __init__(self, model, decay: float = 0.999)
```

Обёртка для управления EMA версией модели.

**Пример:**

```python
from transformers.ema import EMAModel

ema_model = EMAModel(model, decay=0.999)

# Обучение
for batch in dataloader:
    outputs = ema_model.model(batch)
    loss.backward()
    optimizer.step()
    ema_model.update()  # Обновляем EMA

# Для инференса используем EMA веса
with ema_model.use_ema():
    outputs = ema_model.model(inputs)
```

---

### Утилиты

#### compute_optimal_decay

```python
def compute_optimal_decay(
    total_steps: int,
    target_half_life_steps: int = None
) -> float
```

Вычисляет оптимальный decay на основе длины обучения.

```python
from transformers.ema import compute_optimal_decay

decay = compute_optimal_decay(total_steps=10000)
print(f"Recommended decay: {decay}")  # ~0.9993
```

#### print_ema_info

```python
def print_ema_info(decay: float, total_steps: int)
```

Выводит информацию о конфигурации EMA.

```python
from transformers.ema import print_ema_info

print_ema_info(decay=0.999, total_steps=10000)
# ==================================================
# EMA CONFIGURATION
# ==================================================
# Decay:                    0.999
# Half-life:                693 steps
# ...
```

---

## 🎯 Рекомендации по decay

| Длина обучения | Рекомендуемый decay |
|----------------|---------------------|
| 1,000 шагов | 0.99 |
| 10,000 шагов | 0.999 |
| 100,000 шагов | 0.9999 |

**Правило:** Чем длиннее обучение, тем выше decay.

---

## 💾 Сохранение и загрузка EMA

```python
import torch

# Сохранение
ema_state = ema_callback.get_ema_state()
torch.save(ema_state, "ema_weights.pt")

# Загрузка
ema_state = torch.load("ema_weights.pt")
ema_callback.load_ema_state(ema_state)
ema_callback.apply_ema(model)
```

---

## ⚠️ Важные замечания

1. **Память:** EMA требует ~2x памяти для хранения весов
2. **GPU:** EMA state хранится на том же устройстве что и модель
3. **Распределённое обучение:** EMA обновляется на каждом процессе

---

## 📚 Теоретическая база

EMA основан на **Polyak averaging** (1990):

```
θ_ema = β × θ_ema + (1-β) × θ_current
```

Где:
- `θ_ema` — EMA веса
- `θ_current` — текущие веса
- `β` — decay (обычно 0.999)

Используется в:
- Stable Diffusion
- DALL-E
- Большинстве image generation моделей
