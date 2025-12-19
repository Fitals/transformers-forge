# 🔨 Transformers Forge

**Независимый форк библиотеки Transformers с улучшениями качества обучения**

[![Version](https://img.shields.io/badge/version-1.0.7-blue.svg)](CHANGELOG.md)
[![Tests](https://github.com/Fitals/transformers-forge/actions/workflows/forge-unit-tests.yml/badge.svg)](https://github.com/Fitals/transformers-forge/actions/workflows/forge-unit-tests.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)](https://python.org)

---

## 📋 О проекте

**Transformers Forge** — "Кузница трансформеров". Независимый форк с:
- 🔥 **EMA** — улучшение качества +1-3% (проверено)
- ⚡ **Layer Utils** — заморозка слоёв, LP-LoRA стиль
- 📦 **Training Presets** — готовые конфиги для SFT/LoRA/QLoRA/DPO
- 📊 **Training Monitor** — мониторинг GPU, градиентов
- 🐛 **7 Bug Fixes** — критические исправления

---

## 🚀 Быстрый старт

```bash
# Клонировать репозиторий
git clone https://github.com/transformers-forge/transformers-forge.git
cd transformers-forge

# Установить
pip install -e .
```

### Проверка установки

```python
import transformers
print(transformers.__version__)
# Output: 1.0.0
```

---

## 🔥 Главные возможности

### 1. EMA — Улучшение качества +1-3%

```python
from transformers import Trainer
from transformers.ema import EMACallback

trainer = Trainer(
    model=model,
    args=args,
    callbacks=[EMACallback(decay=0.999)]  # +1-3% quality!
)
trainer.train()

# Применить EMA веса
ema_callback.apply_ema(model)
```

### 2. Layer Utils — Заморозка слоёв

```python
from transformers import freeze_first_n_layers, get_frozen_percentage

# LP-LoRA стиль: заморозить 50% слоёв
freeze_first_n_layers(model, n=16)
print(f"Frozen: {get_frozen_percentage(model):.1f}%")  # ~50%
```

### 3. Training Presets — Готовые конфиги

```python
from transformers import get_preset

# Быстрый старт с LoRA
preset = get_preset("qlora", lora_r=32)
training_args = preset.get_training_args()
lora_config = preset.get_lora_config()
bnb_config = preset.get_bnb_config()
```

### 4. Training Monitor — Мониторинг

```python
from transformers import TrainingMonitor, MonitorCallback

trainer = Trainer(
    model=model,
    callbacks=[MonitorCallback(check_gradients=True)]
)
```

---

## 🐛 Исправленные баги

| # | Issue | Проблема |
|---|-------|----------|
| 1 | #42925 | TvpConfig `type_vocab_size` |
| 2 | #42910 | Qwen2VL `size` parameter |
| 3 | #42679 | ConditionalDetr segmentation |
| 4 | #42722 | OneFormer device sync |
| 5 | #42890 | SAM HQ reproducibility |
| 6 | #42759 | Siglip `hidden_states` |
| 7 | #42762 | **GenerationConfig override** ⭐ |

📖 **Подробности:** [CHANGELOG.md](CHANGELOG.md)

---

## 📊 Сравнение с оригиналом

| Возможность | Transformers | Transformers Forge |
|-------------|--------------|-------------------|
| EMA для качества | ❌ | ✅ +1-3% |
| Layer freezing | Вручную | ✅ Автоматизировано |
| Training presets | ❌ | ✅ SFT/LoRA/QLoRA/DPO |
| Bug fixes | Ждать PR | ✅ Сразу |

---

## 📁 Новые модули

```
src/transformers/
├── ema.py               # EMA для улучшения качества
├── layer_utils.py       # Заморозка слоёв
├── training_presets.py  # Готовые конфиги
├── training_monitor.py  # Мониторинг обучения
```

---

## 🤝 Вклад в проект

Приветствуются:
- 🐛 Сообщения о багах
- 💡 Новые идеи
- 🔧 Pull requests

См. [CONTRIBUTING_RU.md](CONTRIBUTING_RU.md)

---

## 📜 Лицензия

Apache License 2.0

---

## 👤 Автор

**Самад Абдулаев (Фиталс)**
- 📧 Email: usnul.noxil@gmail.com
- 💡 Автор идеи и разработчик Transformers Forge

---

## 🙏 Благодарности

- [Hugging Face](https://huggingface.co/) — за базовую библиотеку Transformers
- Community — за идеи и тестирование

---

**🔨 Transformers Forge v1.0.0 — куём лучшее!**

*Created by Самад Абдулаев (Fitals)*
