# Interactive Model Manager

**Модуль:** `transformers.interactive`

Интерактивная консоль для управления моделями и настройки fine-tuning.

---

## 📊 Зачем нужен Interactive Manager?

| Проблема | Решение |
|----------|---------|
| Много ручных настроек | Wizard ведёт по шагам |
| Запутаться в структуре | Сканирует папки автоматически |
| Не знаю какой формат датасета | Показывает примеры и валидирует |

---

## 🔧 Быстрый старт

```python
from transformers.interactive import InteractiveModelManager

# Создаём менеджер
manager = InteractiveModelManager(
    models_dir="./models",
    datasets_dir="./datasets"
)

# Запускаем интерактивную сессию
manager.run()
```

---

## 📖 API Reference

### InteractiveModelManager

```python
class InteractiveModelManager:
    def __init__(
        self,
        models_dir: str = "./models",
        datasets_dir: str = "./datasets"
    )
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `models_dir` | str | "./models" | Директория с HuggingFace моделями |
| `datasets_dir` | str | "./datasets" | Директория с датасетами |

---

### Основные методы

#### scan_models

```python
def scan_models(self) -> List[ModelInfo]
```

Сканирует директорию моделей.

```python
models = manager.scan_models()
for model in models:
    print(f"{model.name}: {model.size_gb:.2f} GB")
```

---

#### scan_datasets

```python
def scan_datasets(self) -> List[DatasetInfo]
```

Сканирует директорию датасетов.

```python
datasets = manager.scan_datasets()
for ds in datasets:
    print(f"{ds.name}: {ds.num_lines} строк")
```

---

#### validate_dataset

```python
def validate_dataset(
    self,
    dataset_path: str,
    max_check: int = 100
) -> ValidationResult
```

Валидирует формат датасета.

```python
result = manager.validate_dataset("./data/train.jsonl")

if result.valid:
    print("✅ Датасет валиден!")
else:
    for error in result.errors:
        print(f"❌ {error}")
```

---

#### run

```python
def run(self)
```

Запускает интерактивную сессию.

---

### Вспомогательные классы

#### ModelInfo

```python
@dataclass
class ModelInfo:
    name: str               # Имя модели
    path: str               # Путь к модели
    size_gb: float          # Размер в GB
    model_type: str         # Тип модели
    num_parameters: str     # Кол-во параметров (1.5B)
    has_tokenizer: bool     # Есть токенизатор
    has_safetensors: bool   # Есть safetensors
    has_pytorch: bool       # Есть pytorch_model.bin
```

#### DatasetInfo

```python
@dataclass
class DatasetInfo:
    name: str      # Имя датасета
    path: str      # Путь
    size_mb: float # Размер в MB
    num_lines: int # Количество строк
    format: str    # Формат (jsonl, json)
```

#### ValidationResult

```python
@dataclass
class ValidationResult:
    valid: bool              # Валиден ли датасет
    total_lines: int         # Всего строк
    valid_lines: int         # Валидных строк
    errors: List[str]        # Ошибки
    warnings: List[str]      # Предупреждения
    sample_line: str         # Пример строки
```

---

## 💡 Полный пример

```python
from transformers.interactive import InteractiveModelManager

# 1. Создаём менеджер
manager = InteractiveModelManager(
    models_dir="/path/to/models",
    datasets_dir="/path/to/datasets"
)

# 2. Сканируем модели
models = manager.scan_models()
print(f"Найдено {len(models)} моделей")

# 3. Сканируем датасеты
datasets = manager.scan_datasets()
print(f"Найдено {len(datasets)} датасетов")

# 4. Валидируем датасет
if datasets:
    result = manager.validate_dataset(datasets[0].path)
    print(f"Датасет {datasets[0].name}:")
    print(f"  Валиден: {result.valid}")
    print(f"  Строк: {result.total_lines}")

# 5. Запускаем wizard
manager.run()
```

---

## 📋 Формат датасета

Interactive Manager работает с ChatML JSONL:

```jsonl
{"messages": [{"role": "system", "content": "You are helpful assistant."}, {"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]}
```

Каждая строка должна содержать:
- `messages` — массив сообщений
- Каждое сообщение: `role` (system/user/assistant) + `content`

---

## 🎯 Fine-Tune Wizard

Wizard помогает настроить fine-tuning:

1. **Выбор модели** — из списка найденных
2. **Проверка зависимостей** — trl, peft, datasets
3. **Сканирование датасетов** — валидация формата
4. **Выбор датасета** — с показом примеров
5. **Настройка параметров** — preset или manual
6. **Запуск обучения**

---

## ⚠️ Важные замечания

1. **GGUF не поддерживается** — для fine-tuning нужен HuggingFace формат
2. **GPU рекомендуется** — CPU будет очень медленным
3. **Формат ChatML** — другие форматы требуют конвертации

---

## 🆕 Добавлено в v1.0.9

- Полноценная интерактивная консоль
- Fine-tune Wizard с валидацией
- Автоустановка зависимостей
- Поддержка presets (SFT, LoRA, QLoRA)
