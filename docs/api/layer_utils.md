# Layer Utils

**Модуль:** `transformers.layer_utils`

Утилиты для безопасного управления слоями модели: заморозка, разморозка, анализ.

---

## 📊 Зачем нужен Layer Utils?

| Проблема | Решение |
|----------|---------|
| Памяти не хватает на все слои | Заморозить часть слоёв |
| LP-LoRA требует заморозку | `freeze_first_n_layers()` |
| Нужно понять структуру модели | `print_layer_status()` |

**Экономия памяти:** до 50%+ при заморозке половины слоёв

---

## 🔧 Быстрый старт

```python
from transformers import AutoModelForCausalLM
from transformers import freeze_first_n_layers, get_frozen_percentage

model = AutoModelForCausalLM.from_pretrained("model_name")

# Заморозить первые 16 слоёв (LP-LoRA стиль)
freeze_first_n_layers(model, n=16)

# Проверить
print(f"Frozen: {get_frozen_percentage(model):.1f}%")
```

---

## 📖 API Reference

### Функции заморозки

#### freeze_model

```python
def freeze_model(model) -> int
```

Заморозить все параметры модели.

```python
from transformers import freeze_model

frozen_count = freeze_model(model)
print(f"Frozen {frozen_count:,} parameters")
```

---

#### freeze_first_n_layers

```python
def freeze_first_n_layers(model, n: int) -> int
```

Заморозить первые N слоёв. **Рекомендуется для LP-LoRA.**

```python
from transformers import freeze_first_n_layers

# Заморозить первые 16 слоёв
freeze_first_n_layers(model, 16)
```

---

#### freeze_last_n_layers

```python
def freeze_last_n_layers(model, n: int) -> int
```

Заморозить последние N слоёв.

---

#### freeze_except_last_n

```python
def freeze_except_last_n(model, n: int) -> int
```

Заморозить всё кроме последних N слоёв.

```python
from transformers import freeze_except_last_n

# Обучать только последние 4 слоя
freeze_except_last_n(model, n=4)
```

---

#### freeze_embeddings

```python
def freeze_embeddings(model) -> int
```

Заморозить слои эмбеддингов. Предотвращает "забывание" знаний.

```python
from transformers import freeze_embeddings

freeze_embeddings(model)
```

---

#### freeze_by_name

```python
def freeze_by_name(
    model,
    patterns: Union[str, List[str]],
    case_sensitive: bool = True
) -> int
```

Заморозить параметры по regex паттерну.

```python
from transformers import freeze_by_name

# Заморозить все attention слои
freeze_by_name(model, [".*attention.*", ".*attn.*"])
```

---

### Функции разморозки

#### unfreeze_model

```python
def unfreeze_model(model) -> int
```

Разморозить все параметры.

---

### Функции анализа

#### get_trainable_params

```python
def get_trainable_params(model) -> int
```

Подсчитать обучаемые параметры.

```python
from transformers import get_trainable_params

trainable = get_trainable_params(model)
print(f"Trainable: {trainable:,}")
```

---

#### get_frozen_percentage

```python
def get_frozen_percentage(model) -> float
```

Получить процент замороженных параметров.

```python
from transformers import get_frozen_percentage

pct = get_frozen_percentage(model)
print(f"Frozen: {pct:.1f}%")
```

---

#### get_num_layers

```python
def get_num_layers(model) -> int
```

Получить количество слоёв в модели.

```python
from transformers import get_num_layers

num = get_num_layers(model)
print(f"Model has {num} layers")
```

---

#### print_layer_status

```python
def print_layer_status(model, show_all: bool = False)
```

Вывести таблицу статуса слоёв.

```python
from transformers import print_layer_status

print_layer_status(model)
# ========================================
# LAYER STATUS
# ========================================
# Layer 0-15:  FROZEN  (1.2B params)
# Layer 16-31: ACTIVE  (1.2B params)
# ----------------------------------------
# Embeddings:  FROZEN
# LM Head:     ACTIVE
# ========================================
```

---

### GradualUnfreezer

```python
class GradualUnfreezer:
    def __init__(
        self,
        model,
        total_epochs: int,
        unfreeze_embeddings_at: int = None,
        freeze_embeddings: bool = True,
        verbose: bool = True
    )
```

Постепенная разморозка слоёв для лучшего transfer learning.

```python
from transformers import GradualUnfreezer

unfreezer = GradualUnfreezer(model, total_epochs=10)

for epoch in range(10):
    unfreezer.step(epoch)  # Размораживает слои постепенно
    train_one_epoch(model, ...)
```

---

### setup_lp_lora_style

```python
def setup_lp_lora_style(
    model,
    freeze_ratio: float = 0.5,
    freeze_embed: bool = True
) -> int
```

Быстрая настройка LP-LoRA стиля.

```python
from transformers import setup_lp_lora_style

# Заморозить 50% слоёв + эмбеддинги
setup_lp_lora_style(model, freeze_ratio=0.5)
```

---

### get_memory_savings_estimate

```python
def get_memory_savings_estimate(model) -> Dict[str, float]
```

Оценить экономию памяти от заморозки.

```python
from transformers import get_memory_savings_estimate

savings = get_memory_savings_estimate(model)
print(f"Gradient memory saved: {savings['gradient_saved_gb']:.2f} GB")
print(f"Optimizer memory saved: {savings['optimizer_saved_gb']:.2f} GB")
```

---

## 🎯 Типичные сценарии

### LP-LoRA стиль

```python
from transformers import setup_lp_lora_style

setup_lp_lora_style(model, freeze_ratio=0.5)
# Результат: первые 50% слоёв заморожены
```

### Обучение только последних слоёв

```python
from transformers import freeze_except_last_n

freeze_except_last_n(model, n=4)
# Результат: только последние 4 слоя обучаются
```

### Постепенная разморозка

```python
from transformers import GradualUnfreezer

unfreezer = GradualUnfreezer(model, total_epochs=10)
# Эпоха 1: только lm_head
# Эпоха 5: половина слоёв
# Эпоха 10: все слои
```

---

## ⚠️ Важные замечания

1. **Безопасность:** Все функции только изменяют `requires_grad`, не модифицируют веса
2. **Обратимость:** Можно в любой момент вызвать `unfreeze_model()`
3. **Совместимость:** Работает с любой PyTorch моделью
