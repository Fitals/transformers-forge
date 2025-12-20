# Flash Mode — Ускоренное обучение

**Модуль:** `transformers.flash_mode`  
**Добавлено в:** v1.1.3

Flash Mode — режим ускоренного обучения, реализующий технологию **Adaptive Layer Activity Spectrum (ALAS)**.

---

## 📊 Ожидаемое ускорение

| Компонент | Вклад |
|-----------|-------|
| ALAS | 15-25% |
| Sample Weighting | 5-10% |
| Conservative GCA | 3-5% |
| **Итого** | **1.3-1.5x** |

---

## 🚀 Быстрый старт

```python
from transformers import Trainer, TrainingArguments
from transformers.flash_mode import FlashConfig, FlashModeCallback

# Конфигурация Flash Mode
config = FlashConfig(
    enable_alas=True,
    enable_sample_weighting=True,
    enable_gca=True,
)

# Добавляем callback в Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[FlashModeCallback(config)],
)

trainer.train()
```

---

## 🔧 Компоненты

### 1. ALAS (Adaptive Layer Activity Spectrum)

Динамически определяет уровень активности каждого слоя и решает: честный backward или интерполяция градиента.

**Как работает:**
1. Вычисляет `heat` — насколько активен слой (gradient magnitude / weight magnitude)
2. Вычисляет `stability` — насколько предсказуемы градиенты
3. Определяет `activity_level` ∈ [0.3, 1.0]
4. Стохастически решает: `random() < activity_level` → backward, иначе интерполяция

**Интерполяция градиентов:**
- Взвешенное среднее с экспоненциальным затуханием
- Учёт тренда (последние 2 градиента)
- Небольшой шум для exploration

```python
from transformers.flash_mode.alas import ALASTracker

tracker = ALASTracker(
    min_activity=0.3,       # Минимум 30% backward
    honest_check_every=20,  # Полный backward каждые 20 шагов
)
```

### 2. Sample Weighting

Снижает вес примеров с низким loss (уже "выученных").

```python
from transformers.flash_mode.sample_weighter import SampleWeighter

weighter = SampleWeighter(
    min_weight=0.3,           # Минимальный вес
    low_loss_threshold=0.5,   # Порог "выученного"
)

# Вычисляем веса
weights = weighter.compute_weights(per_sample_losses)
```

### 3. Conservative GCA

Консервативная экстраполяция градиентов — виртуальные optimizer steps когда градиенты согласованы.

```python
from transformers.flash_mode.gca import ConservativeGCA

gca = ConservativeGCA(
    coherence_threshold=0.95,  # Очень высокий порог
    max_virtual_steps=2,       # Максимум 2 виртуальных шага
    checkpoint_every=5,        # Частые проверки
)
```

---

## ⚙️ FlashConfig

```python
@dataclass
class FlashConfig:
    # Компоненты
    enable_alas: bool = True
    enable_sample_weighting: bool = True
    enable_gca: bool = True
    
    # ALAS параметры
    min_activity: float = 0.3
    honest_check_every: int = 20
    activity_decay: float = 0.95
    
    # Sample Weighting
    min_sample_weight: float = 0.3
    low_loss_threshold: float = 0.5
    
    # GCA параметры
    coherence_threshold: float = 0.95
    max_virtual_steps: int = 2
    gca_checkpoint_every: int = 5
    
    # Safety
    warmup_steps: int = 100
    loss_spike_threshold: float = 1.1
    auto_disable_on_spike: bool = True
    
    # Logging
    verbose: bool = True
    log_every: int = 50
```

---

## 🛡️ Safety механизмы

| Механизм | Описание |
|----------|----------|
| `min_activity = 0.3` | Слой никогда не замораживается полностью |
| `honest_check_every = 20` | Регулярная валидация реальными градиентами |
| `loss_spike_threshold = 1.1` | Авто-отключение при росте loss на 10% |
| `warmup_steps = 100` | Flash Mode не активен первые 100 шагов |

---

## 📈 Статистика

После обучения Flash Mode выводит статистику:

```
╔══════════════════════════════════════════════════════════════════════╗
║  ⚡ FLASH MODE — STATISTICS                                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  Effective Speedup: 1.38x                                            ║
║                                                                      ║
║  Components:                                                         ║
║    • ALAS Savings: 23.5%                                             ║
║    • GCA Virtual Steps: 8.2%                                         ║
║    • Sample Weight Avg: 0.847                                        ║
║                                                                      ║
║  Total Steps: 1000                                                   ║
║  Loss Spikes: 0 | Auto-Disables: 0                                   ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## ⚠️ Когда использовать

✅ **Рекомендуется:**
- Длительное обучение (> 1000 шагов)
- Стабильные гиперпараметры
- Достаточно памяти для истории градиентов

❌ **Не рекомендуется:**
- Очень короткое обучение
- Экспериментальные настройки
- Критически важные модели (лучше стандартное обучение)

---

## 🔬 Научная основа

Flash Mode основан на нескольких наблюдениях:

1. **Слои учатся с разной скоростью** — ранние слои стабилизируются быстрее
2. **Примеры с низким loss дают малый градиент** — меньше информации
3. **Согласованные градиенты указывают на стабильную фазу** — можно экстраполировать

---

## 📚 API Reference

### FlashModeCallback

```python
class FlashModeCallback(TrainerCallback):
    def __init__(
        self,
        config: Optional[FlashConfig] = None,
        verbose: bool = True,
    )
```

### ALASTracker

```python
class ALASTracker:
    def initialize(self, model: nn.Module)
    def update_metrics(self, model: nn.Module)
    def should_compute_backward(self, layer_name: str) -> bool
    def get_stats(self) -> Dict[str, Any]
```

### SampleWeighter

```python
class SampleWeighter:
    def compute_weights(self, losses: torch.Tensor) -> torch.Tensor
    def get_stats(self) -> dict
```

### ConservativeGCA

```python
class ConservativeGCA:
    def update(self, gradient, loss) -> Tuple[bool, int]
    def get_stats(self) -> dict
```
