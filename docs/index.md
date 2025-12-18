# 🔨 Transformers Forge — Документация

**Версия:** 1.0.2  
**Автор:** Самад Абдулаев (Фиталс)

---

## 📚 Содержание

### Быстрый старт
- [Установка](tutorials/quickstart.md)
- [Первые шаги](tutorials/quickstart.md#первые-шаги)

### API Reference
- [EMA (Exponential Moving Average)](api/ema.md) — улучшение качества +1-3%
- [Layer Utils](api/layer_utils.md) — заморозка слоёв, LP-LoRA
- [Training Presets](api/training_presets.md) — готовые конфиги SFT/LoRA/QLoRA/DPO
- [Training Monitor](api/training_monitor.md) — мониторинг обучения

### Tutorials
- [Быстрый старт](tutorials/quickstart.md)
- [Использование EMA](tutorials/ema_guide.md)
- [Fine-tuning с Transformers Forge](tutorials/finetuning.md)

### Research
- [📚 Теория и Практика](RESEARCH.md) — научное обоснование технологий

### Examples
- [examples/](../examples/) — готовые примеры кода

---

## 🚀 Быстрая установка

```bash
git clone https://github.com/Fitals/transformers-forge.git
cd transformers-forge
pip install -e .
```

## ✅ Проверка

```python
import transformers
print(transformers.__version__)  # 1.0.0
```

---

## 🔥 Основные возможности

| Модуль | Описание | Польза |
|--------|----------|--------|
| **EMA** | Сглаживание весов | +1-3% качества |
| **Layer Utils** | Заморозка слоёв | Экономия памяти 50%+ |
| **Training Presets** | Готовые конфиги | Быстрый старт |
| **Training Monitor** | Мониторинг | Отладка обучения |

---

## 📞 Контакты

- **Автор:** Самад Абдулаев (Фиталс)
- **Email:** usnul.noxil@gmail.com
- **GitHub:** [github.com/Fitals/transformers-forge](https://github.com/Fitals/transformers-forge)
