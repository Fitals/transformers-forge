# Training Presets

**Модуль:** `transformers.training_presets`

Готовые конфигурации для типичных сценариев обучения LLM.

---

## 📊 Зачем нужны Presets?

| Проблема | Решение |
|----------|---------|
| Долго подбирать гиперпараметры | Готовые проверенные конфиги |
| Запутаться в настройках LoRA | Пресет уже настроен |
| Не знаю с чего начать | Выбери preset и запускай |

---

## 🔧 Быстрый старт

```python
from transformers import get_preset

# Получить preset для QLoRA
preset = get_preset("qlora")

# Получить готовые аргументы
training_args = preset.get_training_args()
lora_config = preset.get_lora_config()
bnb_config = preset.get_bnb_config()
```

---

## 📖 Доступные Presets

| Preset | Описание | Память | Скорость |
|--------|----------|--------|----------|
| `sft` | Supervised Fine-Tuning | Высокая | Быстро |
| `lora` | LoRA fine-tuning | Средняя | Средне |
| `qlora` | QLoRA (4-bit) | Низкая | Медленнее |
| `dpo` | Direct Preference Optimization | Высокая | Средне |
| `memory_efficient` | Минимум памяти | Очень низкая | Медленно |

---

## 📖 API Reference

### get_preset

```python
def get_preset(
    name: str,
    **overrides
) -> BasePreset
```

Получить preset по имени с возможностью переопределения параметров.

```python
from transformers import get_preset

# Базовый preset
preset = get_preset("qlora")

# С переопределениями
preset = get_preset("qlora", lora_r=64, learning_rate=1e-4)
```

---

### Методы Preset

Каждый preset имеет методы:

| Метод | Возвращает | Описание |
|-------|------------|----------|
| `get_training_args()` | TrainingArguments | Аргументы для Trainer |
| `get_args_dict()` | Dict | Словарь параметров (без accelerate) *(v1.1.0)* |
| `get_lora_config()` | LoraConfig | Конфиг LoRA (если применимо) |
| `get_bnb_config()` | BitsAndBytesConfig | Конфиг квантизации (если применимо) |
| `print_info()` | None | Вывести информацию о preset |

---

## 🎯 Подробное описание Presets

### SFT Preset

```python
preset = get_preset("sft")
```

**Для:** Полное fine-tuning без LoRA.

**Параметры по умолчанию:**
```python
{
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,  # Авто-определяется
    "gradient_checkpointing": True,
}
```

---

### LoRA Preset

```python
preset = get_preset("lora")
```

**Для:** Fine-tuning с LoRA адаптерами.

**LoRA параметры:**
```python
{
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
}
```

---

### QLoRA Preset

```python
preset = get_preset("qlora")
```

**Для:** 4-bit fine-tuning с минимальным потреблением памяти.

**BnB Config:**
```python
{
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
}
```

---

### DPO Preset

```python
preset = get_preset("dpo")
```

**Для:** Direct Preference Optimization (выравнивание с предпочтениями).

**DPO параметры:**
```python
{
    "beta": 0.1,
    "learning_rate": 5e-7,
    "max_length": 1024,
    "max_prompt_length": 512,
}
```

---

### Memory Efficient Preset

```python
preset = get_preset("memory_efficient")
```

**Для:** Максимальная экономия памяти.

**Особенности:**
- 4-bit квантизация
- Gradient checkpointing
- Маленький batch size
- Высокий gradient accumulation

---

## 💡 Полный пример

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers import get_preset
from peft import get_peft_model

# 1. Получаем preset
preset = get_preset("qlora", lora_r=32)

# 2. Загружаем модель с квантизацией
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=preset.get_bnb_config()
)

# 3. Добавляем LoRA
model = get_peft_model(model, preset.get_lora_config())

# 4. Создаём Trainer
trainer = Trainer(
    model=model,
    args=preset.get_training_args(),
    train_dataset=dataset,
)

# 5. Обучаем
trainer.train()
```

---

## 🛠 Кастомизация

### Переопределение параметров

```python
preset = get_preset(
    "qlora",
    learning_rate=1e-4,
    lora_r=64,
    lora_alpha=128,
    num_train_epochs=5
)
```

### Создание своего preset

```python
from transformers.training_presets import BasePreset

class MyPreset(BasePreset):
    name = "my_preset"
    description = "Мой кастомный preset"
    
    def get_training_args(self):
        return TrainingArguments(
            output_dir="./output",
            learning_rate=1e-5,
            # ... ваши параметры
        )
```

---

## ⚠️ Важные замечания

1. **Авто-определение:** Presets автоматически определяют GPU/CPU и bf16/fp16
2. **PEFT зависимость:** LoRA/QLoRA presets требуют установленную библиотеку `peft`
3. **BnB зависимость:** QLoRA требует `bitsandbytes`
