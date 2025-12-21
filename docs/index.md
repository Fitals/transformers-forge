# 🔨 Transformers Forge — Документация

**Версия:** 1.1.4  
**Автор:** Самад Абдулаев (Фиталс)

---

## 📚 Содержание

### Быстрый старт
- [Установка](tutorials/quickstart.md)
- [Первые шаги](tutorials/quickstart.md#первые-шаги)

### API Reference
- [EMA (Exponential Moving Average)](api/ema.md) — улучшение качества +1-3%*
- [Layer Utils](api/layer_utils.md) — заморозка слоёв, LP-LoRA, Smart Freeze
- [Training Presets](api/training_presets.md) — готовые конфиги SFT/LoRA/QLoRA/DPO/CPT/DoRA/ORPO
- [Training Monitor](api/training_monitor.md) — мониторинг обучения
- [Interactive Manager](api/interactive.md) — интерактивная консоль *(v1.0.9)*
- [LR Finder](api/lr_finder.md) — автоматический подбор learning rate *(v1.1.1)*
- [Flash Mode](api/flash_mode.md) — ускоренное обучение 1.3-1.5x *(v1.1.3)*
- [Dataset Utils](api/dataset_utils.md) — анализ датасетов *(v1.1.4)*
- [Adaptive Loss](api/adaptive_loss.md) — адаптивные функции потерь *(v1.1.4)*

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
print(transformers.__version__)  # 1.1.4
```

---

## 🔥 Основные возможности

| Модуль | Описание | Польза |
|--------|----------|--------|
| **EMA** | Сглаживание весов | +1-3% качества* |
| **Layer Utils** | Заморозка слоёв + Smart Freeze | Экономия памяти 50%+ |
| **Training Presets** | 9 готовых конфигов | Быстрый старт |
| **Training Monitor** | Мониторинг | Отладка обучения |
| **Interactive Manager** | Wizard для fine-tuning | Упрощённая настройка |
| **LR Finder** | Подбор learning rate | Оптимальный LR за 2 мин |
| **Flash Mode** | Ускоренное обучение | 1.3-1.5x быстрее |
| **Dataset Utils** | Анализ датасетов | Рекомендации по обучению |
| **Adaptive Loss** | Focal loss, response-only | Фокус на сложных примерах |

---

## 📞 Контакты

- **Автор:** Самад Абдулаев (Фиталс)
- **Email:** usnul.noxil@gmail.com
- **GitHub:** [github.com/Fitals/transformers-forge](https://github.com/Fitals/transformers-forge)
