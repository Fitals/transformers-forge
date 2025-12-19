# Learning Rate Finder

**Модуль:** `transformers.lr_finder`

Автоматический подбор оптимального learning rate на основе метода Leslie Smith (2015).

---

## 📊 Зачем нужен LR Finder?

| Проблема | Что происходит | Решение |
|----------|----------------|---------|
| LR слишком высокий | Loss взрывается, модель ломается | LR Finder покажет точку взрыва |
| LR слишком низкий | Модель не учится, время потрачено | LR Finder найдёт зону обучения |
| Угадывание LR | Много экспериментов | Автоматический подбор за 2 минуты |

---

## 🔧 Быстрый старт

```python
from transformers.lr_finder import find_optimal_lr

# Найти оптимальный LR за одну строку
optimal_lr = find_optimal_lr(model, train_dataloader)
print(f"Используй LR: {optimal_lr}")
```

---

## 📖 API Reference

### LRFinder

```python
class LRFinder:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[callable] = None,
        device: str = "auto"
    )
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model` | nn.Module | — | PyTorch модель |
| `train_dataloader` | DataLoader | — | Данные для обучения |
| `optimizer` | Optimizer | AdamW | Оптимизатор (опционально) |
| `criterion` | callable | None | Функция loss (опционально) |
| `device` | str | "auto" | Устройство: "auto", "cuda", "cpu" |

---

### Методы

#### find()

```python
def find(
    self,
    min_lr: float = 1e-8,
    max_lr: float = 1e-1,
    num_steps: int = 100,
    smooth_factor: float = 0.98,
    divergence_threshold: float = 4.0,
    suggestion_method: str = "steepest_gradient"
) -> LRFinderResult
```

Запускает поиск оптимального LR.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `min_lr` | float | 1e-8 | Минимальный LR для теста |
| `max_lr` | float | 1e-1 | Максимальный LR для теста |
| `num_steps` | int | 100 | Количество шагов |
| `smooth_factor` | float | 0.98 | Сглаживание loss (0-1) |
| `divergence_threshold` | float | 4.0 | Порог остановки при взрыве |
| `suggestion_method` | str | "steepest_gradient" | Метод выбора LR |

**Возвращает:** `LRFinderResult`

---

#### plot()

```python
def plot(
    self,
    output_path: Optional[str] = None,
    log_scale: bool = True,
    show_suggestion: bool = True
) -> Optional[str]
```

Строит график loss vs learning rate.

```python
finder.plot("lr_curve.png")
```

---

#### reset()

```python
def reset(self)
```

Восстанавливает оригинальные веса модели.

---

### LRFinderResult

```python
@dataclass
class LRFinderResult:
    optimal_lr: float       # Рекомендуемый LR
    min_lr: float           # Мин тестируемый LR
    max_lr: float           # Макс тестируемый LR
    num_steps: int          # Количество шагов
    lrs: List[float]        # Все LR
    losses: List[float]     # Loss на каждом шаге
    smoothed_losses: List[float]  # Сглаженные
    best_lr_idx: int        # Индекс лучшего LR
    suggestion_method: str  # Метод выбора
```

---

## 💡 Полный пример

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.lr_finder import LRFinder
from torch.utils.data import DataLoader
from datasets import load_dataset

# 1. Загружаем модель
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. Готовим данные
dataset = load_dataset("text", data_files="train.txt", split="train")

def tokenize(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

dataloader = DataLoader(tokenized, batch_size=4, shuffle=True)

# 3. Запускаем LR Finder
finder = LRFinder(model, dataloader)
result = finder.find(num_steps=100)

print(f"Оптимальный LR: {result.optimal_lr:.2e}")

# 4. Сохраняем график
finder.plot("lr_finder.png")

# 5. Используем найденный LR
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    learning_rate=result.optimal_lr,  # Используем найденный LR
    num_train_epochs=3,
)
```

---

## 🔬 Как это работает

1. **Экспоненциальный рост LR**: от `min_lr` до `max_lr`
2. **Запись loss**: на каждом шаге
3. **Сглаживание**: exponential moving average
4. **Анализ**: поиск точки максимального снижения loss
5. **Рекомендация**: LR чуть ниже этой точки

```
Loss
  │
  │\
  │ \          ← Оптимальный LR здесь
  │  \        /
  │   \______/
  │           \
  │            \  ← Divergence
  └──────────────────── LR
    1e-7      1e-4     1e-1
```

---

## ⚠️ Важные замечания

1. **Веса восстанавливаются** — после теста модель в исходном состоянии
2. **Нужен GPU** — на CPU работает, но медленно
3. **Batch size важен** — при OOM уменьшите batch size
4. **100 шагов достаточно** — больше не нужно

---

## 📚 Научное обоснование

- **Leslie Smith (2015)** — "Cyclical Learning Rates for Training Neural Networks"
- **Ссылка:** https://arxiv.org/abs/1506.01186

Метод используется в:
- FastAI
- PyTorch Lightning
- Keras
- И теперь в Transformers Forge!

---

## 🆕 Добавлено в v1.1.1

- Полноценный LR Finder
- Экспорт через `from transformers import LRFinder`
- График loss vs LR
- Автоматическое восстановление весов
