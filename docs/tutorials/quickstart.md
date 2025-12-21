# 🚀 Быстрый старт

Это руководство поможет начать работу с Transformers Forge за 5 минут.

---

## 📦 Установка

```bash
git clone https://github.com/Fitals/transformers-forge.git
cd transformers-forge
pip install -e .
```

---

## ✅ Проверка установки

```python
import transformers
print(f"Transformers Forge v{transformers.__version__}")

# Должно вывести: Transformers Forge v1.1.4
```

---

## 🔥 Первые шаги

### 1. EMA для улучшения качества

```python
from transformers import Trainer, TrainingArguments
from transformers.ema import EMACallback

# Создаём EMA callback
ema_callback = EMACallback(decay=0.999)

# Добавляем в Trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./output"),
    train_dataset=dataset,
    callbacks=[ema_callback]
)

# Обучаем
trainer.train()

# Применяем EMA веса (лучше качество!)
ema_callback.apply_ema(model)
model.save_pretrained("./best_model")
```

### 2. Заморозка слоёв для экономии памяти

```python
from transformers import freeze_first_n_layers, get_frozen_percentage

# Заморозить первые 16 слоёв
freeze_first_n_layers(model, n=16)

print(f"Frozen: {get_frozen_percentage(model):.1f}%")
# Результат: ~50% памяти экономится
```

### 3. Готовые конфиги для QLoRA

```python
from transformers import get_preset

preset = get_preset("qlora")

# Получи всё что нужно
training_args = preset.get_training_args()
lora_config = preset.get_lora_config()
bnb_config = preset.get_bnb_config()
```

### 4. Мониторинг обучения

```python
from transformers import print_model_info, MonitorCallback

# Посмотреть информацию о модели
print_model_info(model)

# Добавить мониторинг
trainer = Trainer(
    model=model,
    args=args,
    callbacks=[MonitorCallback(check_gradients=True)]
)
```

---

## 🎯 Что дальше?

- [Гайд по EMA](ema_guide.md) — подробно об улучшении качества
- [Fine-tuning гайд](finetuning.md) — полный пример обучения
- [API Reference](../api/) — документация всех функций

---

## 💬 Помощь

- **GitHub Issues:** [github.com/Fitals/transformers-forge/issues](https://github.com/Fitals/transformers-forge/issues)
- **Email:** usnul.noxil@gmail.com
