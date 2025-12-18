# Training Monitor

**Модуль:** `transformers.training_monitor`

Утилиты для мониторинга и отладки обучения моделей.

---

## 📊 Зачем нужен Training Monitor?

| Проблема | Решение |
|----------|---------|
| Не понимаю сколько памяти нужно | `estimate_model_memory()` |
| Градиенты взрываются/исчезают | `check_gradient_health()` |
| Хочу видеть прогресс | `MonitorCallback` |

---

## 🔧 Быстрый старт

```python
from transformers import Trainer
from transformers import MonitorCallback, print_model_info

# Вывести информацию о модели
print_model_info(model)

# Добавить мониторинг в Trainer
trainer = Trainer(
    model=model,
    args=args,
    callbacks=[MonitorCallback(check_gradients=True)]
)
```

---

## 📖 API Reference

### Функции анализа модели

#### count_parameters

```python
def count_parameters(
    model,
    trainable_only: bool = False
) -> int
```

Подсчёт параметров модели.

```python
from transformers import count_parameters

total = count_parameters(model)
trainable = count_parameters(model, trainable_only=True)
print(f"Total: {total:,}, Trainable: {trainable:,}")
```

---

#### format_param_count

```python
def format_param_count(count: int) -> str
```

Форматирование числа параметров (1.5B, 7M, 3K).

```python
from transformers import format_param_count

print(format_param_count(1_500_000_000))  # "1.50B"
print(format_param_count(7_000_000))       # "7.00M"
```

---

#### get_parameter_breakdown

```python
def get_parameter_breakdown(model) -> List[Dict]
```

Детальная разбивка параметров по слоям.

```python
from transformers import get_parameter_breakdown

breakdown = get_parameter_breakdown(model)
for layer in breakdown[:5]:
    print(f"{layer['name']}: {layer['params']:,} ({layer['dtype']})")
```

---

#### estimate_model_memory

```python
def estimate_model_memory(
    model,
    batch_size: int = 1,
    sequence_length: int = 512,
    precision: str = "fp16"
) -> Dict[str, float]
```

Оценка потребления памяти.

```python
from transformers import estimate_model_memory

memory = estimate_model_memory(model, batch_size=4, sequence_length=2048)
print(f"Parameters: {memory['parameters_gb']:.2f} GB")
print(f"Gradients: {memory['gradients_gb']:.2f} GB")
print(f"Optimizer: {memory['optimizer_gb']:.2f} GB")
print(f"Total: {memory['total_estimated_gb']:.2f} GB")
```

---

#### print_model_info

```python
def print_model_info(
    model,
    show_breakdown: bool = False
)
```

Красивый вывод информации о модели.

```python
from transformers import print_model_info

print_model_info(model)
# ================================================
# MODEL INFO
# ================================================
# Total parameters:     7,000,000,000 (7.00B)
# Trainable parameters: 1,000,000 (1.00M)
# Frozen parameters:    6,999,000,000
# Trainable ratio:      0.01%
# ================================================
# Memory Estimation (FP16):
# Parameters:           13.04 GB
# Gradients:            0.00 GB (frozen excluded)
# Optimizer (AdamW):    26.08 GB
# ================================================
```

---

### Функции проверки здоровья

#### check_gradient_health

```python
def check_gradient_health(
    model,
    warn_threshold: float = 1.0,
    error_threshold: float = 10.0
) -> Dict
```

Проверка здоровья градиентов.

```python
from transformers import check_gradient_health

# После backward() но до optimizer.step()
health = check_gradient_health(model)

if not health["healthy"]:
    print("⚠️ Gradient issues detected!")
    for issue in health["issues"]:
        print(f"  - {issue}")
```

**Возвращает:**
- `healthy`: bool — всё ли нормально
- `issues`: list — список проблем
- `stats`: dict — статистика (min, max, mean, std)

---

### GradientStats

```python
@dataclass
class GradientStats:
    min_grad: float
    max_grad: float
    mean_grad: float
    std_grad: float
    num_zero: int
    num_nan: int
    num_inf: int
```

Статистика градиентов.

---

### TrainingMetrics

```python
@dataclass  
class TrainingMetrics:
    step: int
    loss: float
    learning_rate: float
    gradient_norm: float
    gpu_memory_used: float
    gpu_memory_total: float
```

Метрики обучения для логирования.

---

### TrainingMonitor

```python
class TrainingMonitor:
    def __init__(self, model, check_every: int = 100)
```

Полный мониторинг обучения.

```python
from transformers import TrainingMonitor

monitor = TrainingMonitor(model)

# Получить summary
summary = monitor.get_model_summary()
print(f"Total params: {summary['total_parameters']}")
print(f"Trainable: {summary['trainable_parameters']}")
```

---

### MonitorCallback

```python
class MonitorCallback(TrainerCallback):
    def __init__(
        self,
        print_model_summary: bool = True,
        log_gpu_memory: bool = True,
        check_gradients: bool = False,
        gradient_check_steps: int = 100
    )
```

Callback для интеграции с `Trainer`.

```python
from transformers import Trainer
from transformers import MonitorCallback

trainer = Trainer(
    model=model,
    args=args,
    callbacks=[
        MonitorCallback(
            print_model_summary=True,
            log_gpu_memory=True,
            check_gradients=True
        )
    ]
)
```

**Функции callback:**
- При старте — выводит информацию о модели
- Каждые N шагов — проверяет градиенты (если включено)
- Логирует GPU memory в wandb/tensorboard

---

### GPU мониторинг

#### get_gpu_memory_info

```python
def get_gpu_memory_info() -> Dict[str, float]
```

Информация о памяти GPU.

```python
from transformers.training_monitor import get_gpu_memory_info

gpu_info = get_gpu_memory_info()
print(f"Used: {gpu_info['used_gb']:.2f} GB")
print(f"Free: {gpu_info['free_gb']:.2f} GB")
print(f"Total: {gpu_info['total_gb']:.2f} GB")
```

---

## 💡 Полный пример

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import (
    MonitorCallback,
    print_model_info,
    estimate_model_memory
)

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained("model_name")

# Предварительный анализ
print_model_info(model, show_breakdown=True)

memory = estimate_model_memory(model, batch_size=4, sequence_length=2048)
print(f"\nEstimated memory: {memory['total_estimated_gb']:.2f} GB")

# Обучение с мониторингом
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./output"),
    train_dataset=dataset,
    callbacks=[
        MonitorCallback(
            check_gradients=True,
            gradient_check_steps=50
        )
    ]
)

trainer.train()
```

---

## ⚠️ Важные замечания

1. **GPU:** Функции GPU работают только с CUDA
2. **Производительность:** `check_gradients` добавляет небольшой overhead
3. **Совместимость:** Работает с любой PyTorch моделью
