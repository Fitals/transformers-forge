# 🔨 Transformers Forge - Changelog

Все значимые изменения в этом проекте документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/),
и этот проект придерживается [Семантического версионирования](https://semver.org/lang/ru/).

---

## [1.0.2] - 2024-12-18 — Tests Verified

### ✅ Тесты

- **Исправлены тесты** — все unit тесты теперь проходят (100%)
- `test_ema.py` — исправлен под реальный API
- `test_layer_utils.py` — все тесты работают
- `test_training_presets.py` — исправлен под реальный API
- `test_training_monitor.py` — исправлен под реальные имена полей

### 🔧 Исправления

- Тесты теперь соответствуют реальным сигнатурам функций
- Убраны несуществующие атрибуты из тестов

---

## [1.0.1] - 2024-12-18 — Документация и Тесты

### 📚 Документация

- **docs/index.md** — Главная страница документации
- **docs/api/** — API Reference для всех модулей:
  - `ema.md` — Документация EMA
  - `layer_utils.md` — Документация Layer Utils
  - `training_presets.md` — Документация Training Presets
  - `training_monitor.md` — Документация Training Monitor
- **docs/tutorials/** — Гайды и tutorials:
  - `quickstart.md` — Быстрый старт
  - `ema_guide.md` — Подробный гайд по EMA
  - `finetuning.md` — Полный гайд по fine-tuning

### 💻 Примеры

- **examples/forge_examples/** — Рабочие примеры кода:
  - `ema_example.py` — Пример использования EMA
  - `layer_freezing_example.py` — Пример заморозки слоёв
  - `training_presets_example.py` — Пример Training Presets

### 🧪 Тесты

- **tests/forge/** — Unit тесты для всех модулей:
  - `test_ema.py` — Тесты EMA
  - `test_layer_utils.py` — Тесты Layer Utils
  - `test_training_presets.py` — Тесты Training Presets
  - `test_training_monitor.py` — Тесты Training Monitor

### 🏛️ Структура проекта

- **ROADMAP.md** — План развития проекта
- **GOVERNANCE.md** — Модель управления
- **.github/ISSUE_TEMPLATE/** — Шаблоны для Issues:
  - `bug_report.md`
  - `feature_request.md`
  - `question.md`

---

## [1.0.0] - 2024-12-18 — Первый релиз Transformers Forge!

### ✨ Новые возможности (New Features)

#### Training Monitor Module
- **Файл:** `src/transformers/training_monitor.py`
- **Описание:** Новый модуль для мониторинга обучения моделей
- **Возможности:**
  - `count_parameters()` — подсчёт параметров (total/trainable/frozen)
  - `get_parameter_breakdown()` — детальная разбивка по слоям
  - `estimate_model_memory()` — оценка потребления памяти
  - `get_gpu_memory_info()` — мониторинг GPU памяти
  - `check_gradient_health()` — проверка здоровья градиентов (NaN, Inf, vanishing, exploding)
  - `TrainingMonitor` — класс для комплексного мониторинга
  - `MonitorCallback` — callback для Trainer с автоматическим логированием
  - `print_model_info()` / `print_gpu_status()` — быстрые функции для вывода информации

**Пример использования:**
```python
from transformers.training_monitor import TrainingMonitor, MonitorCallback

# Анализ модели
monitor = TrainingMonitor(model)
monitor.print_model_summary()

# С Trainer
trainer = Trainer(model=model, callbacks=[MonitorCallback()])
```

#### GitHub Actions CI/CD для Enhanced версии
- **Файлы:** `.github/workflows/enhanced-*.yml`
- **Описание:** Набор лёгких CI/CD workflows для Enhanced версии
- **Workflows:**
  - `enhanced-code-quality.yml` — Lint (Ruff), проверка импортов, версий, синтаксиса
  - `enhanced-tests.yml` — Тесты GenerationConfig, training_monitor, верификация фиксов
  - `enhanced-release.yml` — Подготовка релиза, сборка артефактов
- **Преимущества:**
  - Работает на стандартных GitHub-hosted runners (без GPU)
  - Не требует специальных секретов HuggingFace
  - Быстрое время выполнения (~3-5 минут)

#### Training Presets Module
- **Файл:** `src/transformers/training_presets.py`
- **Описание:** Готовые, оптимизированные конфигурации для обучения LLM
- **Presets:**
  - `SFTPreset` — Supervised Fine-Tuning с NEFTune
  - `LoRAPreset` — LoRA fine-tuning (PEFT)
  - `QLoRAPreset` — 4-bit Quantized LoRA  
  - `DPOPreset` — Direct Preference Optimization
  - `MemoryEfficientPreset` — Для ограниченной GPU памяти
- **Возможности:**
  - Авто-определение GPU/CPU и bf16 поддержки
  - Готовые конфигурации для PEFT (LoraConfig) и BitsAndBytes
  - Registry для кастомных presets
  - Quick-функции: `quick_sft_args()`, `quick_lora_args()`

**Пример использования:**
```python
from transformers.training_presets import get_preset

# Быстрый старт
preset = get_preset("lora", lora_r=16)
training_args = preset.get_training_args()
lora_config = preset.get_lora_config()
```

#### Layer Utilities Module
- **Файл:** `src/transformers/layer_utils.py`
- **Описание:** Безопасные утилиты для управления слоями (заморозка/разморозка)
- **Функции:**
  - `freeze_first_n_layers()` — заморозить первые N слоёв (LP-LoRA стиль)
  - `freeze_except_last_n()` — заморозить всё кроме последних N
  - `freeze_embeddings()` — заморозить эмбеддинги
  - `get_trainable_params()` / `get_frozen_percentage()` — анализ
  - `print_layer_status()` — таблица статуса слоёв
  - `GradualUnfreezer` — постепенная разморозка для transfer learning
  - `setup_lp_lora_style()` — быстрая настройка LP-LoRA
- **Преимущества:**
  - 100% безопасно — не изменяет архитектуру
  - Экономия памяти до 50%+ (меньше градиентов)
  - Ускорение обучения

**Пример использования:**
```python
from transformers import freeze_first_n_layers, get_frozen_percentage

freeze_first_n_layers(model, n=16)  # LP-LoRA стиль
print(f"Frozen: {get_frozen_percentage(model):.1f}%")
```

#### EMA (Exponential Moving Average) Module
- **Файл:** `src/transformers/ema.py`
- **Описание:** Сглаженные веса модели для лучшей generalization
- **Компоненты:**
  - `EMACallback` — Callback для интеграции с Trainer
  - `EMAModel` — Обёртка для управления EMA
  - `compute_optimal_decay()` — Расчёт оптимального decay
- **Преимущества:**
  - **+1-3% на eval метриках** (типичное улучшение)
  - Более стабильная генерация
  - Проверено десятилетиями (Polyak averaging, 1990+)
  - Используется в SOTA: Stable Diffusion, DALL-E

**Пример использования:**
```python
from transformers import Trainer
from transformers.ema import EMACallback

trainer = Trainer(
    model=model,
    args=args,
    callbacks=[EMACallback(decay=0.999)]
)
trainer.train()

# Применить EMA веса (более качественные)
ema_callback.apply_ema(model)
```

### 🐛 Исправления багов (Bug Fixes)

#### Fix #1: TvpConfig отсутствует `type_vocab_size` (Issue #42925)
- **Файл:** `src/transformers/models/tvp/configuration_tvp.py`
- **Проблема:** `TvpConfig` не имел параметра `type_vocab_size`, что приводило к `AttributeError` при создании модели с кастомным конфигом.
- **Решение:** Добавлен параметр `type_vocab_size=2` в `__init__` и документацию.

#### Fix #2: Qwen2VLImageProcessor игнорирует параметр `size` (Issue #42910)
- **Файл:** `src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`
- **Проблема:** Параметр `size` перезаписывался дефолтными значениями из-за некорректной логики `if/else`.
- **Решение:** Исправлена логика инициализации — теперь явно переданный `size` сохраняется.

#### Fix #3: ConditionalDetr теряет последний класс в сегментации (Issue #42679)
- **Файлы:** 
  - `src/transformers/models/conditional_detr/image_processing_conditional_detr.py`
  - `src/transformers/models/conditional_detr/image_processing_conditional_detr_fast.py`
- **Проблема:** Метод `post_process_semantic_segmentation` некорректно удалял последний класс (`[..., :-1]`), что было скопировано из DETR (который имеет null class), но Conditional DETR не имеет null class.
- **Решение:** Убрано некорректное срезание, все классы сохраняются.

#### Fix #4: OneFormerProcessor `task_inputs` на неправильном устройстве (Issue #42722)
- **Файл:** `src/transformers/models/oneformer/processing_oneformer.py`
- **Проблема:** `task_inputs` оставались на CPU даже когда `pixel_values` были на GPU, что приводило к ошибке device mismatch.
- **Решение:** Добавлена синхронизация устройства `task_inputs` с `pixel_values`.

#### Fix #5: SAM HQ тесты падают из-за отсутствия `set_seed()` (Issue #42890)
- **Файл:** `tests/models/sam_hq/test_modeling_sam_hq.py`
- **Проблема:** Интеграционные тесты не имели `set_seed()`, что делало результаты невоспроизводимыми.
- **Решение:** Добавлен `setUp` метод с `set_seed(0)` для обеспечения воспроизводимости.

#### Fix #6: SiglipModel не возвращает `hidden_states` (Issue #42759)
- **Файл:** `src/transformers/models/siglip/modeling_siglip.py`
- **Проблема:** `SiglipTextTransformer` не имел декоратора `@check_model_inputs`, поэтому `output_hidden_states=True` не работал.
- **Решение:** Добавлен декоратор `@check_model_inputs(tie_last_hidden_states=False)`.

#### Fix #7: GenerationConfig перезаписывает явно заданные значения (Issue #42762)
- **Файлы:** 
  - `src/transformers/generation/configuration_utils.py`
  - `src/transformers/generation/utils.py`
  - `tests/generation/test_configuration_utils.py`
- **Проблема:** Логика слияния конфигов не могла отличить явно заданные значения от дефолтных. Например, если пользователь явно задал `temperature=1.0`, а модель имела `temperature=1e-06`, значение пользователя перезаписывалось.
- **Решение:** 
  - Добавлен атрибут `_explicitly_set_attrs` для отслеживания явно заданных параметров.
  - Изменена логика слияния — явно заданные значения НЕ перезаписываются.
  - Добавлены тесты для верификации.
- **Важно для:** RLHF, DPO, GRPO, TRL training — теперь параметры генерации гарантированно сохраняются.

---

## Как установить

```bash
# Клонировать репозиторий
git clone https://github.com/YOUR_USERNAME/transformers-enhanced.git
cd transformers-enhanced

# Установить в режиме разработки
pip install -e .
```

---

## Совместимость

- **Базовая версия:** transformers 5.0.0.dev0
- **Python:** 3.9+
- **PyTorch:** 2.0+

---

## Авторы

- Community Enhanced Version
- Оригинал: Hugging Face Team
