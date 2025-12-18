# 🎓 Fine-tuning с Transformers Forge

Полное руководство по fine-tuning LLM с использованием всех возможностей Transformers Forge.

---

## 📋 Содержание

1. [Подготовка](#подготовка)
2. [QLoRA fine-tuning](#qlora-fine-tuning)
3. [Оптимизация памяти](#оптимизация-памяти)
4. [Улучшение качества с EMA](#улучшение-качества-с-ema)
5. [Полный пример](#полный-пример)

---

## Подготовка

### Установка зависимостей

```bash
pip install -e .
pip install peft bitsandbytes accelerate datasets
```

### Загрузка данных

```python
from datasets import load_dataset

dataset = load_dataset("your_dataset")
```

---

## QLoRA Fine-tuning

### Шаг 1: Использование preset

```python
from transformers import get_preset

# Получаем готовый preset
preset = get_preset("qlora", lora_r=32, learning_rate=2e-4)

# Выводим информацию
preset.print_info()
```

### Шаг 2: Загрузка модели с квантизацией

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=preset.get_bnb_config(),
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
```

### Шаг 3: Добавление LoRA

```python
from peft import prepare_model_for_kbit_training, get_peft_model

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, preset.get_lora_config())
```

### Шаг 4: Обучение

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=preset.get_training_args(),
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()
```

---

## Оптимизация памяти

### Заморозка слоёв (LP-LoRA стиль)

```python
from transformers import setup_lp_lora_style, get_memory_savings_estimate

# Заморозить 50% слоёв
setup_lp_lora_style(model, freeze_ratio=0.5)

# Оценить экономию
savings = get_memory_savings_estimate(model)
print(f"Gradient memory saved: {savings['gradient_saved_gb']:.2f} GB")
```

### Анализ модели перед обучением

```python
from transformers import print_model_info, estimate_model_memory

# Информация о модели
print_model_info(model, show_breakdown=True)

# Оценка памяти
memory = estimate_model_memory(model, batch_size=4, sequence_length=2048)
print(f"Estimated total: {memory['total_estimated_gb']:.2f} GB")
```

---

## Улучшение качества с EMA

### Добавление EMA

```python
from transformers.ema import EMACallback

ema_callback = EMACallback(decay=0.999)

trainer = Trainer(
    model=model,
    args=preset.get_training_args(),
    train_dataset=dataset["train"],
    callbacks=[ema_callback]  # Добавляем EMA
)

trainer.train()

# Применяем EMA веса
ema_callback.apply_ema(model)
```

---

## Полный пример

Полный скрипт `train_with_forge.py`:

```python
"""
Transformers Forge: Полный пример fine-tuning
"""

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer,
    get_preset,
    print_model_info,
    setup_lp_lora_style,
    MonitorCallback
)
from transformers.ema import EMACallback
from peft import prepare_model_for_kbit_training, get_peft_model

# =============================================================================
# Конфигурация
# =============================================================================

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET_NAME = "your_dataset"
OUTPUT_DIR = "./output"

# =============================================================================
# 1. Получаем preset
# =============================================================================

preset = get_preset(
    "qlora",
    lora_r=32,
    lora_alpha=64,
    learning_rate=2e-4,
    num_train_epochs=3
)

print("📦 Preset configuration:")
preset.print_info()

# =============================================================================
# 2. Загружаем модель
# =============================================================================

print(f"\n📥 Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=preset.get_bnb_config(),
    device_map="auto"
)

# =============================================================================
# 3. Подготовка модели
# =============================================================================

print("\n🔧 Preparing model...")

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, preset.get_lora_config())

# LP-LoRA стиль: заморозить часть слоёв
setup_lp_lora_style(model, freeze_ratio=0.5)

# Информация о модели
print_model_info(model)

# =============================================================================
# 4. Загрузка данных
# =============================================================================

print(f"\n📚 Loading dataset: {DATASET_NAME}")

dataset = load_dataset(DATASET_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

dataset = dataset.map(tokenize, batched=True)

# =============================================================================
# 5. Настройка callbacks
# =============================================================================

ema_callback = EMACallback(decay=0.999)
monitor_callback = MonitorCallback(check_gradients=True)

# =============================================================================
# 6. Trainer
# =============================================================================

training_args = preset.get_training_args()
training_args.output_dir = OUTPUT_DIR

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset.get("validation"),
    tokenizer=tokenizer,
    callbacks=[ema_callback, monitor_callback]
)

# =============================================================================
# 7. Обучение
# =============================================================================

print("\n🚀 Starting training...")
trainer.train()

# =============================================================================
# 8. Сохранение
# =============================================================================

print("\n💾 Saving model with EMA weights...")

# Применяем EMA веса (лучше качество)
ema_callback.apply_ema(model)

# Сохраняем
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

print(f"\n✅ Done! Model saved to {OUTPUT_DIR}/final_model")
```

---

## 🎯 Чеклист fine-tuning

- [ ] Выбрать правильный preset (sft/lora/qlora/dpo)
- [ ] Проверить оценку памяти `estimate_model_memory()`
- [ ] Заморозить часть слоёв для экономии памяти
- [ ] Добавить EMA для улучшения качества
- [ ] Добавить мониторинг градиентов
- [ ] Применить EMA веса перед сохранением

---

## 💡 Советы

1. **Начните с QLoRA** — минимальный расход памяти
2. **Используйте EMA** — бесплатные +1-3% качества
3. **Замораживайте слои** — LP-LoRA даёт хорошие результаты
4. **Мониторьте градиенты** — ловите проблемы рано
