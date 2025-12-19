# Copyright 2024 Community Enhanced Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Interactive Model Manager for Transformers Forge.

Provides an interactive console interface for:
- Scanning and selecting local models
- Scanning and validating datasets
- Running inference (chat)
- Setting up fine-tuning

Usage:
    from transformers.interactive import InteractiveModelManager
    
    manager = InteractiveModelManager(
        models_dir="./models",
        datasets_dir="./datasets"
    )
    manager.run()

Or from command line:
    python -m transformers.interactive --models ./models --datasets ./datasets
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import logging

logger = logging.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelInfo:
    """Information about a discovered model."""
    name: str
    path: str
    size_gb: float
    model_type: str = "Unknown"
    num_parameters: Optional[str] = None
    has_tokenizer: bool = False
    has_safetensors: bool = False
    has_pytorch: bool = False


@dataclass
class DatasetInfo:
    """Information about a discovered dataset."""
    name: str
    path: str
    size_mb: float
    num_lines: int = 0
    format: str = "unknown"  # jsonl, json, csv


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    valid: bool
    total_lines: int = 0
    valid_lines: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sample_line: Optional[str] = None


# =============================================================================
# Interactive Model Manager
# =============================================================================


class InteractiveModelManager:
    """
    Interactive console interface for model management.
    
    Provides a user-friendly way to:
    - Browse and select local models
    - Validate and select datasets
    - Run inference or set up fine-tuning
    
    Args:
        models_dir: Directory containing HuggingFace models
        datasets_dir: Directory containing training datasets
        
    Example:
        >>> from transformers.interactive import InteractiveModelManager
        >>> manager = InteractiveModelManager(
        ...     models_dir="./models",
        ...     datasets_dir="./datasets"
        ... )
        >>> manager.run()
    """
    
    def __init__(
        self,
        models_dir: str = "./models",
        datasets_dir: str = "./datasets"
    ):
        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        
        self.selected_model: Optional[ModelInfo] = None
        self.selected_dataset: Optional[DatasetInfo] = None
    
    # =========================================================================
    # Model Scanning
    # =========================================================================
    
    def scan_models(self) -> List[ModelInfo]:
        """
        Scan models directory for HuggingFace models.
        
        Looks for directories containing config.json (HF format).
        Excludes GGUF files as they don't support fine-tuning.
        
        Returns:
            List of ModelInfo objects for discovered models.
        """
        models = []
        
        if not self.models_dir.exists():
            return models
        
        for item in self.models_dir.iterdir():
            if not item.is_dir():
                continue
            
            config_path = item / "config.json"
            if not config_path.exists():
                continue
            
            # Skip if contains GGUF files
            gguf_files = list(item.glob("*.gguf"))
            if gguf_files:
                continue
            
            # Get model info
            model_info = self._get_model_info(item)
            if model_info:
                models.append(model_info)
        
        return sorted(models, key=lambda m: m.name)
    
    def _get_model_info(self, model_path: Path) -> Optional[ModelInfo]:
        """Extract model information from directory."""
        try:
            # Read config
            config_path = model_path / "config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # Calculate size
            total_size = sum(
                f.stat().st_size for f in model_path.rglob("*") if f.is_file()
            )
            size_gb = total_size / (1024 ** 3)
            
            # Get model type
            model_type = config.get("model_type", "Unknown")
            architectures = config.get("architectures", [])
            if architectures:
                model_type = architectures[0]
            
            # Check for files
            has_tokenizer = (model_path / "tokenizer.json").exists() or \
                           (model_path / "tokenizer_config.json").exists()
            has_safetensors = bool(list(model_path.glob("*.safetensors")))
            has_pytorch = bool(list(model_path.glob("*.bin")))
            
            # Estimate parameters
            num_params = None
            if "num_parameters" in config:
                num_params = self._format_params(config["num_parameters"])
            elif "hidden_size" in config and "num_hidden_layers" in config:
                # Rough estimate
                h = config["hidden_size"]
                l = config["num_hidden_layers"]
                v = config.get("vocab_size", 32000)
                estimated = l * 12 * h * h + v * h
                num_params = self._format_params(estimated)
            
            return ModelInfo(
                name=model_path.name,
                path=str(model_path),
                size_gb=size_gb,
                model_type=model_type,
                num_parameters=num_params,
                has_tokenizer=has_tokenizer,
                has_safetensors=has_safetensors,
                has_pytorch=has_pytorch
            )
        except Exception as e:
            logger.warning(f"Failed to read model {model_path}: {e}")
            return None
    
    def _format_params(self, count: int) -> str:
        """Format parameter count (1.5B, 7M, etc.)."""
        if count >= 1e9:
            return f"{count / 1e9:.1f}B"
        elif count >= 1e6:
            return f"{count / 1e6:.1f}M"
        else:
            return f"{count / 1e3:.1f}K"
    
    # =========================================================================
    # Dataset Scanning & Validation
    # =========================================================================
    
    def scan_datasets(self) -> List[DatasetInfo]:
        """
        Scan datasets directory for training files.
        
        Looks for .jsonl, .json files.
        
        Returns:
            List of DatasetInfo objects.
        """
        datasets = []
        
        if not self.datasets_dir.exists():
            return datasets
        
        for ext in ["*.jsonl", "*.json"]:
            for file_path in self.datasets_dir.glob(ext):
                if file_path.is_file():
                    info = self._get_dataset_info(file_path)
                    if info:
                        datasets.append(info)
        
        return sorted(datasets, key=lambda d: d.name)
    
    def _get_dataset_info(self, file_path: Path) -> Optional[DatasetInfo]:
        """Extract dataset information."""
        try:
            size_mb = file_path.stat().st_size / (1024 ** 2)
            
            # Count lines
            num_lines = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for _ in f:
                    num_lines += 1
            
            # Determine format
            fmt = "jsonl" if file_path.suffix == ".jsonl" else "json"
            
            return DatasetInfo(
                name=file_path.name,
                path=str(file_path),
                size_mb=size_mb,
                num_lines=num_lines,
                format=fmt
            )
        except Exception as e:
            logger.warning(f"Failed to read dataset {file_path}: {e}")
            return None
    
    def validate_dataset(self, dataset_path: str, max_check: int = 100) -> ValidationResult:
        """
        Validate dataset format and content.
        
        Checks:
        - Valid JSON/JSONL format
        - Presence of 'messages' field
        - Each message has 'role' and 'content'
        - Valid roles (system, user, assistant)
        
        Args:
            dataset_path: Path to dataset file
            max_check: Maximum lines to check (for large files)
            
        Returns:
            ValidationResult with errors and warnings.
        """
        errors = []
        warnings = []
        valid_lines = 0
        total_lines = 0
        sample_line = None
        
        valid_roles = {"system", "user", "assistant"}
        
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    total_lines += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    # Only check first max_check lines
                    if i >= max_check:
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i+1}: Invalid JSON - {e}")
                        continue
                    
                    # Check for messages field
                    if "messages" not in data:
                        errors.append(f"Line {i+1}: Missing 'messages' field")
                        continue
                    
                    messages = data["messages"]
                    if not isinstance(messages, list):
                        errors.append(f"Line {i+1}: 'messages' must be a list")
                        continue
                    
                    if len(messages) == 0:
                        warnings.append(f"Line {i+1}: Empty messages list")
                        continue
                    
                    # Check each message
                    line_valid = True
                    for j, msg in enumerate(messages):
                        if not isinstance(msg, dict):
                            errors.append(f"Line {i+1}, msg {j}: Not a dict")
                            line_valid = False
                            continue
                        
                        if "role" not in msg:
                            errors.append(f"Line {i+1}, msg {j}: Missing 'role'")
                            line_valid = False
                        elif msg["role"] not in valid_roles:
                            warnings.append(f"Line {i+1}, msg {j}: Unknown role '{msg['role']}'")
                        
                        if "content" not in msg:
                            errors.append(f"Line {i+1}, msg {j}: Missing 'content'")
                            line_valid = False
                    
                    if line_valid:
                        valid_lines += 1
                        if sample_line is None:
                            sample_line = line[:500]  # First 500 chars
            
        except Exception as e:
            errors.append(f"Failed to read file: {e}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            total_lines=total_lines,
            valid_lines=valid_lines,
            errors=errors[:10],  # Limit errors shown
            warnings=warnings[:5],
            sample_line=sample_line
        )
    
    def show_dataset_example(self):
        """Show example of correct dataset format."""
        print()
        print("=" * 70)
        print("📄 ФОРМАТ ДАТАСЕТА (ChatML JSONL)")
        print("=" * 70)
        print()
        print("Файл: train.jsonl (одна строка = один пример)")
        print()
        print("┌─────────────────────────────────────────────────────────────────────┐")
        print("│ ОБЫЧНЫЙ ФОРМАТ:                                                     │")
        print("├─────────────────────────────────────────────────────────────────────┤")
        print('│ {"messages": [{"role": "system", "content": "Ты помощник."},        │')
        print('│ {"role": "user", "content": "Вопрос"}, {"role": "assistant",        │')
        print('│ "content": "Ответ"}]}                                               │')
        print("└─────────────────────────────────────────────────────────────────────┘")
        print()
        print("┌─────────────────────────────────────────────────────────────────────┐")
        print("│ С REASONING (<think>):                                              │")
        print("├─────────────────────────────────────────────────────────────────────┤")
        print('│ {"messages": [..., {"role": "assistant", "content":                 │')
        print('│ "<think>Размышление...</think>\\n\\nОтвет"}]}                         │')
        print("└─────────────────────────────────────────────────────────────────────┘")
        print()
        print("⚠️  Важно:")
        print("   • Каждая строка — отдельный JSON объект")
        print("   • Обязательные поля: messages, role, content")
        print("   • Роли: system, user, assistant")
        print("=" * 70)
        print()
    
    # =========================================================================
    # UI Methods
    # =========================================================================
    
    def _print_header(self, title: str):
        """Print a formatted header."""
        print()
        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)
    
    def _print_disclaimer(self, title: str, lines: List[str]):
        """Print a disclaimer box."""
        print()
        print("=" * 70)
        print(f"⚠️  {title}")
        print("=" * 70)
        for line in lines:
            print(f"   {line}")
        print("=" * 70)
    
    def show_models_menu(self, models: List[ModelInfo]) -> Optional[ModelInfo]:
        """Display models selection menu."""
        self._print_header("📂 ОБНАРУЖЕННЫЕ МОДЕЛИ")
        
        if not models:
            print()
            print("   ❌ Модели не найдены!")
            print()
            print(f"   Проверьте папку: {self.models_dir.absolute()}")
            print()
            print("   Пример структуры:")
            print("   ./models/")
            print("   └── Qwen2.5-3B/")
            print("       ├── config.json")
            print("       ├── model.safetensors")
            print("       └── tokenizer.json")
            print()
            return None
        
        print(f"\n   Найдено моделей: {len(models)}\n")
        
        for i, model in enumerate(models, 1):
            status = "✅" if model.has_safetensors or model.has_pytorch else "⚠️"
            print(f"   [{i}] {status} {model.name}")
            print(f"       Размер: {model.size_gb:.1f} GB | Тип: {model.model_type}")
            if model.num_parameters:
                print(f"       Параметры: ~{model.num_parameters}")
            print()
        
        print("   [0] ❌ Выход")
        print("=" * 70)
        
        try:
            choice = input("\n   Выберите модель [0-{}]: ".format(len(models))).strip()
            
            if choice == "0" or choice == "":
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                print(f"\n   ✅ Выбрана: {models[idx].name}")
                return models[idx]
            else:
                print("   ❌ Неверный выбор")
                return None
                
        except (ValueError, KeyboardInterrupt, EOFError):
            return None
    
    def show_datasets_menu(self, datasets: List[DatasetInfo]) -> Optional[DatasetInfo]:
        """Display datasets selection menu."""
        self._print_header("📁 ОБНАРУЖЕННЫЕ ДАТАСЕТЫ")
        
        if not datasets:
            print()
            print("   ❌ Датасеты не найдены!")
            print()
            print(f"   Проверьте папку: {self.datasets_dir.absolute()}")
            print()
            print("   Пример структуры:")
            print("   ./datasets/")
            print("   └── train.jsonl")
            print()
            self.show_dataset_example()
            return None
        
        print(f"\n   Найдено датасетов: {len(datasets)}\n")
        
        for i, ds in enumerate(datasets, 1):
            print(f"   [{i}] 📄 {ds.name}")
            print(f"       Размер: {ds.size_mb:.1f} MB | Строк: {ds.num_lines:,} | Формат: {ds.format}")
            print()
        
        print("   [E] 📝 Показать пример формата")
        print("   [0] 🔙 Назад")
        print("=" * 70)
        
        try:
            choice = input("\n   Выберите датасет [0-{}/E]: ".format(len(datasets))).strip().upper()
            
            if choice == "E":
                self.show_dataset_example()
                return self.show_datasets_menu(datasets)  # Show menu again
            
            if choice == "0" or choice == "":
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                selected = datasets[idx]
                
                # Validate dataset
                print(f"\n   ⏳ Проверка датасета...")
                result = self.validate_dataset(selected.path)
                
                if result.valid:
                    print(f"   ✅ Датасет валиден: {result.valid_lines}/{result.total_lines} строк OK")
                else:
                    print(f"   ⚠️ Найдены проблемы:")
                    for err in result.errors[:3]:
                        print(f"      - {err}")
                    
                    proceed = input("\n   Продолжить с этим датасетом? [y/N]: ").strip().lower()
                    if proceed not in ["y", "yes", "да", "д"]:
                        return None
                
                return selected
            else:
                print("   ❌ Неверный выбор")
                return None
                
        except (ValueError, KeyboardInterrupt, EOFError):
            return None
    
    def show_actions_menu(self, model: ModelInfo) -> Optional[str]:
        """Display actions menu for selected model."""
        self._print_header(f"🎯 ДЕЙСТВИЯ: {model.name}")
        
        print()
        print("   Что вы хотите сделать с моделью?")
        print()
        print("   [1] 📊 Анализ модели (summary, параметры)")
        print("   [2] 💬 Запустить чат (inference)")
        print("   [3] 🎯 Fine-tune модели")
        print()
        print("   [0] 🔙 Выбрать другую модель")
        print("=" * 70)
        
        try:
            choice = input("\n   Ваш выбор [0-3]: ").strip()
            
            actions = {
                "1": "analyze",
                "2": "chat", 
                "3": "finetune",
                "0": None
            }
            
            return actions.get(choice, None)
            
        except (KeyboardInterrupt, EOFError):
            return None
    
    def run_analyze(self, model: ModelInfo):
        """Run model analysis."""
        self._print_header(f"📊 АНАЛИЗ: {model.name}")
        
        print()
        print("   ⏳ Загрузка модели для анализа...")
        print()
        
        try:
            from .layer_utils import print_model_summary
            from transformers import AutoModelForCausalLM
            
            loaded_model = AutoModelForCausalLM.from_pretrained(
                model.path,
                trust_remote_code=True,
                device_map="auto"
            )
            
            print_model_summary(loaded_model)
            
            del loaded_model
            
        except Exception as e:
            print(f"   ❌ Ошибка загрузки: {e}")
        
        input("\n   Нажмите Enter для продолжения...")
    
    def run_finetune_wizard(self, model: ModelInfo):
        """Run fine-tuning setup wizard with full validation."""
        # Disclaimer
        self._print_disclaimer(
            "ДИСКЛЕЙМЕР: FINE-TUNING",
            [
                "Fine-tuning изменяет веса модели и требует значительных ресурсов.",
                "",
                "Требования:",
                "• GPU с VRAM >= 8GB (рекомендуется 24GB+)",
                "• Подготовленный датасет в формате ChatML JSONL",
                "• Свободное место на диске (для чекпоинтов)",
                "• Время: от 30 минут до нескольких часов",
                "",
                "🔨 Transformers Forge включает:",
                "• ProgressCallback — красивый прогресс-бар с ETA",
                "• EarlyStoppingCallback — защита от переобучения (Y/N)",
                "• TrainingReportCallback — отчёт training_report.md",
            ]
        )
        
        proceed = input("\n   Продолжить? [Y/n]: ").strip().lower()
        if proceed in ["n", "no", "нет", "н"]:
            return
        
        # Check dependencies first
        self._print_header("🔍 ПРОВЕРКА ЗАВИСИМОСТЕЙ")
        
        missing_deps = []
        
        try:
            import trl
            print("   ✅ trl установлен")
        except ImportError:
            missing_deps.append("trl")
            print("   ❌ trl не установлен")
        
        try:
            import datasets
            print("   ✅ datasets установлен")
        except ImportError:
            missing_deps.append("datasets")
            print("   ❌ datasets не установлен")
        
        try:
            import peft
            print("   ✅ peft установлен")
        except ImportError:
            print("   ⚠️ peft не установлен (LoRA будет недоступен)")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   ✅ GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            else:
                print("   ⚠️ GPU не обнаружен — обучение будет на CPU (очень медленно)")
        except ImportError:
            print("   ⚠️ PyTorch не установлен")
            missing_deps.append("torch")
        
        if missing_deps:
            print()
            print(f"   ❌ Отсутствуют зависимости: {', '.join(missing_deps)}")
            print()
            print("   ┌─────────────────────────────────────────────────────────────────┐")
            print("   │ КАК УСТАНОВИТЬ?                                                 │")
            print("   ├─────────────────────────────────────────────────────────────────┤")
            print("   │  [A] Автоматически (pip install прямо сейчас)                   │")
            print("   │  [M] Вручную (покажем команду, выйдете и установите сами)       │")
            print("   └─────────────────────────────────────────────────────────────────┘")
            
            choice = input("\n   Ваш выбор [A/M]: ").strip().upper()
            
            if choice == "A":
                print()
                print("   ⏳ Устанавливаем зависимости...")
                print(f"      pip install {' '.join(missing_deps)}")
                print()
                
                import subprocess
                result = subprocess.run(
                    ["pip", "install"] + missing_deps,
                    capture_output=False
                )
                
                if result.returncode == 0:
                    print()
                    print("   ✅ Зависимости установлены!")
                    print("   🔄 Перезапустите fine-tune для продолжения.")
                else:
                    print()
                    print("   ❌ Ошибка установки. Попробуйте вручную:")
                    print(f"      pip install {' '.join(missing_deps)}")
                
                input("\n   Нажмите Enter для возврата...")
                return
            else:
                print()
                print("   📝 Выполните в терминале:")
                print()
                print(f"      pip install {' '.join(missing_deps)}")
                print()
                print("   ⚠️ Рекомендуется:")
                print("      1. Нажать Enter для возврата в меню")
                print("      2. Выбрать [0] для выхода")
                print("      3. Установить зависимости в терминале")
                print("      4. Запустить снова")
                input("\n   Нажмите Enter для возврата...")
                return
        
        print()
        print("   ✅ Все зависимости в порядке!")
        
        # Select dataset
        datasets = self.scan_datasets()
        dataset = self.show_datasets_menu(datasets)
        
        if dataset is None:
            return
        
        self.selected_dataset = dataset
        
        # Get training configuration with validation
        config = self._get_training_config_interactive()
        if config is None:
            return
        
        # Final confirmation
        self._print_header("⚙️ ИТОГОВАЯ КОНФИГУРАЦИЯ")
        
        print()
        print(f"   Модель:        {model.name}")
        print(f"   Датасет:       {dataset.name} ({dataset.num_lines:,} примеров)")
        print()
        print(f"   Learning Rate: {config['lr']}")
        print(f"   Batch Size:    {config['batch_size']}")
        print(f"   Epochs:        {config['epochs']}")
        print(f"   LoRA:          {'Да' if config['use_lora'] else 'Нет'}")
        print(f"   Сохранение:    {config['output_dir']}")
        print()
        
        # Show Forge technologies info
        print("   ┌─────────────────────────────────────────────────────────────────┐")
        print("   │ 🔨 ТЕХНОЛОГИИ TRANSFORMERS FORGE                                │")
        print("   ├─────────────────────────────────────────────────────────────────┤")
        print("   │  ✅ ProgressCallback — красивый прогресс-бар с ETA              │")
        print("   │  ✅ EarlyStoppingCallback — авто-стоп при переобучении          │")
        print("   │     (интерактивно спросит Y/N перед остановкой)                 │")
        print("   │  ✅ TrainingReportCallback — отчёт после обучения               │")
        print("   │     (создаст training_report.md с результатами)                 │")
        print("   └─────────────────────────────────────────────────────────────────┘")
        print()
        
        confirm = input("   🚀 Запустить обучение? [Y/n]: ").strip().lower()
        if confirm in ["n", "no", "нет", "н"]:
            print("   ❌ Отменено")
            return
        
        # Run training!
        self._run_finetune_training(model, dataset, config)
    
    def _get_training_config_interactive(self) -> Optional[Dict[str, Any]]:
        """Get training configuration with validation, presets and Auto mode."""
        self._print_header("⚙️ НАСТРОЙКА ПАРАМЕТРОВ")
        
        print()
        print("   ┌─────────────────────────────────────────────────────────────────┐")
        print("   │ 💡 ПОДСКАЗКА                                                    │")
        print("   ├─────────────────────────────────────────────────────────────────┤")
        print("   │  • Введите 'Auto' — автоматические рекомендованные настройки   │")
        print("   │  • Или выбирайте пресеты A/B/C для каждого параметра           │")
        print("   │  • B = рекомендованный (⭐)                                     │")
        print("   └─────────────────────────────────────────────────────────────────┘")
        print()
        
        # Check for Auto mode
        auto_check = input("   Режим настройки [Auto/manual]: ").strip().lower()
        
        if auto_check in ["auto", "а", "авто", ""]:
            print()
            print("   ✅ Выбран режим Auto — используем рекомендованные настройки:")
            print("      • Learning Rate: 2e-5")
            print("      • Batch Size: 4")
            print("      • Epochs: 3")
            print("      • LoRA: Да")
            print("      • Сохранение в: ./output")
            return {
                "lr": 2e-5,
                "batch_size": 4,
                "epochs": 3,
                "use_lora": True,
                "output_dir": "./output"
            }
        
        print()
        print("   📝 Ручная настройка параметров")
        print()
        
        # Learning Rate
        lr = self._get_validated_lr_with_presets()
        if lr is None:
            return None
        
        # Batch Size
        batch_size = self._get_validated_batch_size_with_presets()
        if batch_size is None:
            return None
        
        # Epochs
        epochs = self._get_validated_epochs_with_presets()
        if epochs is None:
            return None
        
        # LoRA
        print()
        print("   LoRA (экономит память):")
        print("      [A] Нет — полное дообучение (требует много VRAM)")
        print("      [B] Да  — LoRA адаптеры (⭐ рекомендуется)")
        use_lora_input = input("   Ваш выбор [A/B]: ").strip().upper()
        use_lora = use_lora_input != "A"
        
        # Output directory
        print()
        print("   Куда сохранить обученную модель?")
        output_dir = input("   Путь [./output]: ").strip()
        if not output_dir:
            output_dir = "./output"
        print(f"   ✅ Модель будет сохранена в: {output_dir}")
        
        return {
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "use_lora": use_lora,
            "output_dir": output_dir
        }
    
    def _get_validated_lr_with_presets(self) -> Optional[float]:
        """Get learning rate with A/B/C presets."""
        print("   Learning Rate:")
        print("      [A] 5e-5  — агрессивный (быстрое обучение)")
        print("      [B] 2e-5  — рекомендованный (⭐)")
        print("      [C] 1e-5  — консервативный (осторожное обучение)")
        print("      Или введите своё значение (например: 3e-5)")
        
        presets = {"a": 5e-5, "b": 2e-5, "c": 1e-5}
        
        max_attempts = 3
        for attempt in range(max_attempts):
            choice = input("   Ваш выбор [A/B/C или число]: ").strip().lower()
            
            if choice in presets:
                lr = presets[choice]
                label = "агрессивный" if choice == "a" else "рекомендованный" if choice == "b" else "консервативный"
                print(f"   ✅ Learning Rate: {lr} ({label})")
                return lr
            
            # Try to parse as number
            try:
                lr = float(choice)
                
                if lr <= 0:
                    print("   ❌ Некорректно: Learning rate должен быть > 0")
                    continue
                
                if lr > 1e-2:
                    print("   ❌ Некорректно: Learning rate слишком высокий (макс. 1e-2)")
                    print("      Возможно вы ошиблись? Используйте пресеты A/B/C")
                    continue
                
                # Warning for unusual values
                if lr > 1e-3:
                    print(f"   ⚠️ Высокий LR ({lr}) — может быть нестабильно")
                    confirm = input("   Продолжить? [y/N]: ").strip().lower()
                    if confirm not in ["y", "yes", "да", "д"]:
                        continue
                
                print(f"   ✅ Learning Rate: {lr} (кастомный)")
                return lr
                
            except ValueError:
                print("   ❌ Введите A, B, C или число (например: 2e-5)")
        
        print("   ❌ Превышено количество попыток")
        return None
    
    def _get_validated_batch_size_with_presets(self) -> Optional[int]:
        """Get batch size with A/B/C presets."""
        print()
        print("   Batch Size:")
        print("      [A] 8   — больше (требует больше VRAM)")
        print("      [B] 4   — рекомендованный (⭐)")
        print("      [C] 2   — меньше (экономит память)")
        print("      Или введите своё значение")
        
        presets = {"a": 8, "b": 4, "c": 2}
        
        max_attempts = 3
        for attempt in range(max_attempts):
            choice = input("   Ваш выбор [A/B/C или число]: ").strip().lower()
            
            if choice in presets:
                bs = presets[choice]
                label = "большой" if choice == "a" else "рекомендованный" if choice == "b" else "экономный"
                print(f"   ✅ Batch Size: {bs} ({label})")
                return bs
            
            try:
                bs = int(choice)
                
                if bs <= 0:
                    print("   ❌ Некорректно: Batch size должен быть > 0")
                    continue
                
                if bs > 64:
                    print("   ❌ Некорректно: Batch size слишком большой (макс. 64)")
                    print("      Используйте пресеты A/B/C")
                    continue
                
                if bs > 16:
                    print(f"   ⚠️ Большой batch ({bs}) — требует много VRAM")
                    confirm = input("   Продолжить? [y/N]: ").strip().lower()
                    if confirm not in ["y", "yes", "да", "д"]:
                        continue
                
                print(f"   ✅ Batch Size: {bs} (кастомный)")
                return bs
                
            except ValueError:
                print("   ❌ Введите A, B, C или число")
        
        print("   ❌ Превышено количество попыток")
        return None
    
    def _get_validated_epochs_with_presets(self) -> Optional[int]:
        """Get epochs with A/B/C presets."""
        print()
        print("   Epochs (эпохи обучения):")
        print("      [A] 5   — больше (дольше, риск переобучения)")
        print("      [B] 3   — рекомендованный (⭐)")
        print("      [C] 1   — меньше (быстро, для больших датасетов)")
        print("      Или введите своё значение")
        
        presets = {"a": 5, "b": 3, "c": 1}
        
        max_attempts = 3
        for attempt in range(max_attempts):
            choice = input("   Ваш выбор [A/B/C или число]: ").strip().lower()
            
            if choice in presets:
                ep = presets[choice]
                label = "интенсивный" if choice == "a" else "рекомендованный" if choice == "b" else "быстрый"
                print(f"   ✅ Epochs: {ep} ({label})")
                return ep
            
            try:
                ep = int(choice)
                
                if ep <= 0:
                    print("   ❌ Некорректно: Epochs должен быть > 0")
                    continue
                
                if ep > 20:
                    print("   ❌ Некорректно: Слишком много эпох (макс. 20)")
                    print("      Используйте пресеты A/B/C")
                    continue
                
                if ep > 10:
                    print(f"   ⚠️ Много эпох ({ep}) — риск переобучения")
                    confirm = input("   Продолжить? [y/N]: ").strip().lower()
                    if confirm not in ["y", "yes", "да", "д"]:
                        continue
                
                print(f"   ✅ Epochs: {ep} (кастомный)")
                return ep
                
            except ValueError:
                print("   ❌ Введите A, B, C или число")
        
        print("   ❌ Превышено количество попыток")
        return None
    
    def _get_validated_lr(self) -> Optional[float]:
        """Get and validate learning rate with disclaimers."""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                lr_input = input("   Learning Rate [2e-5]: ").strip()
                
                if lr_input == "":
                    return 2e-5
                
                lr = float(lr_input)
                
                # Validate
                if lr <= 0:
                    print("   ❌ Некорректно: Learning rate должен быть > 0")
                    continue
                
                if lr > 1e-2:
                    print("   ❌ Некорректно: Learning rate слишком высокий (макс. 1e-2)")
                    print("      Возможно вы ошиблись при вводе?")
                    continue
                
                # Critical values - need confirmation
                if lr > 1e-3:
                    if not self._confirm_critical_setting(
                        setting="Learning Rate",
                        value=str(lr),
                        issue="Высокий LR (> 1e-3) может привести к нестабильному обучению",
                        recommendation="Рекомендуется: 1e-5 до 5e-5",
                        explanation=[
                            "Learning rate определяет размер шага при обновлении весов.",
                            "Слишком высокий LR приводит к:",
                            "  • Хаотичным изменениям весов",
                            "  • Loss может расти вместо снижения",
                            "  • Модель может 'разучиться' и выдавать мусор",
                            "",
                            "Если вы уверены и хотите экспериментировать — продолжайте.",
                        ]
                    ):
                        continue
                
                if lr < 1e-7:
                    if not self._confirm_critical_setting(
                        setting="Learning Rate",
                        value=str(lr),
                        issue="Очень низкий LR (< 1e-7) — обучение будет крайне медленным",
                        recommendation="Рекомендуется: 1e-5 до 5e-5",
                        explanation=[
                            "С таким низким learning rate модель почти не будет учиться.",
                            "Потребуется в 100+ раз больше эпох для того же эффекта.",
                            "",
                            "Если это намеренно для микро-дообучения — продолжайте.",
                        ]
                    ):
                        continue
                
                return lr
                
            except ValueError:
                print("   ❌ Некорректно: Learning rate должен быть числом")
                print("      Пример: 2e-5, 0.00002, 5e-6")
        
        print(f"   ❌ Превышено количество попыток")
        return None
    
    def _get_validated_batch_size(self) -> Optional[int]:
        """Get and validate batch size."""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                bs_input = input("   Batch Size [4]: ").strip()
                
                if bs_input == "":
                    return 4
                
                batch_size = int(bs_input)
                
                if batch_size <= 0:
                    print("   ❌ Некорректно: Batch size должен быть > 0")
                    continue
                
                if batch_size > 128:
                    print("   ❌ Некорректно: Batch size слишком большой (макс. 128)")
                    print("      Возможно вы ошиблись при вводе?")
                    continue
                
                # Critical - large batch
                if batch_size > 32:
                    if not self._confirm_critical_setting(
                        setting="Batch Size",
                        value=str(batch_size),
                        issue="Большой batch size требует много VRAM",
                        recommendation="Рекомендуется: 4-16 для большинства GPU",
                        explanation=[
                            f"Batch size {batch_size} может потребовать 40+ GB VRAM.",
                            "Если памяти не хватит — обучение упадёт с OOM ошибкой.",
                            "",
                            "Совет: используйте gradient_accumulation_steps вместо",
                            "большого batch_size для эффективного увеличения батча.",
                        ]
                    ):
                        continue
                
                return batch_size
                
            except ValueError:
                print("   ❌ Некорректно: Batch size должен быть целым числом")
                print("      Пример: 4, 8, 16")
        
        print(f"   ❌ Превышено количество попыток")
        return None
    
    def _get_validated_epochs(self) -> Optional[int]:
        """Get and validate number of epochs."""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                ep_input = input("   Epochs [3]: ").strip()
                
                if ep_input == "":
                    return 3
                
                epochs = int(ep_input)
                
                if epochs <= 0:
                    print("   ❌ Некорректно: Epochs должен быть > 0")
                    continue
                
                if epochs > 100:
                    print("   ❌ Некорректно: Слишком много эпох (макс. 100)")
                    print("      Возможно вы ошиблись при вводе?")
                    continue
                
                # Critical - many epochs
                if epochs > 10:
                    if not self._confirm_critical_setting(
                        setting="Epochs",
                        value=str(epochs),
                        issue="Много эпох — риск переобучения",
                        recommendation="Рекомендуется: 1-5 для fine-tuning",
                        explanation=[
                            f"{epochs} эпох может привести к переобучению (overfitting).",
                            "Модель запомнит датасет наизусть вместо обобщения.",
                            "Обычно 1-3 эпохи достаточно для fine-tuning.",
                            "",
                            "Если датасет очень большой (100k+ примеров) —",
                            "даже 1 эпоха может быть много.",
                        ]
                    ):
                        continue
                
                return epochs
                
            except ValueError:
                print("   ❌ Некорректно: Epochs должен быть целым числом")
                print("      Пример: 1, 2, 3")
        
        print(f"   ❌ Превышено количество попыток")
        return None
    
    def _confirm_critical_setting(
        self,
        setting: str,
        value: str,
        issue: str,
        recommendation: str,
        explanation: List[str]
    ) -> bool:
        """
        Two-stage confirmation for critical settings.
        
        First asks Y/n, if N shows detailed explanation and asks again.
        """
        # First confirmation
        print()
        print("   ┌─────────────────────────────────────────────────────────────────┐")
        print(f"   │ ⚠️  ВНИМАНИЕ: Специфические настройки                           │")
        print("   ├─────────────────────────────────────────────────────────────────┤")
        print(f"   │  {setting}: {value}")
        print(f"   │  Проблема: {issue}")
        print(f"   │  {recommendation}")
        print("   └─────────────────────────────────────────────────────────────────┘")
        
        first = input("\n   Продолжить с этими настройками? [y/N]: ").strip().lower()
        
        if first in ["y", "yes", "да", "д"]:
            return True
        
        # Second confirmation with detailed explanation
        print()
        print("   ┌─────────────────────────────────────────────────────────────────┐")
        print(f"   │ 📖 ПОДРОБНОЕ ОБЪЯСНЕНИЕ                                         │")
        print("   ├─────────────────────────────────────────────────────────────────┤")
        for line in explanation:
            print(f"   │  {line}")
        print("   └─────────────────────────────────────────────────────────────────┘")
        
        second = input("\n   Вы уверены? Использовать эти настройки? [y/N]: ").strip().lower()
        
        return second in ["y", "yes", "да", "д"]
    
    def _run_finetune_training(self, model: ModelInfo, dataset: DatasetInfo, config: Dict[str, Any]):
        """Actually run the fine-tuning process."""
        self._print_header("🔥 ЗАПУСК ОБУЧЕНИЯ")
        
        print()
        print("   ⏳ Загрузка модели и токенизатора...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from trl import SFTTrainer, SFTConfig
            from datasets import load_dataset
            
            # Import our callbacks
            from .training_monitor import (
                ProgressCallback,
                EarlyStoppingCallback,
                TrainingReportCallback
            )
            
            # Create output directory if needed
            output_dir = Path(config["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"   📁 Папка сохранения: {output_dir.absolute()}")
            
            # Load model
            print(f"   📦 Загрузка {model.name}...")
            
            load_kwargs = {
                "trust_remote_code": True,
            }
            
            # Try to use GPU efficiently
            try:
                import torch
                if torch.cuda.is_available():
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["torch_dtype"] = torch.bfloat16
                    print("   ✅ GPU обнаружен, используем bf16")
            except:
                pass
            
            loaded_model = AutoModelForCausalLM.from_pretrained(
                model.path,
                **load_kwargs
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model.path,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("   ✅ Модель загружена!")
            
            # Load dataset
            print(f"   📁 Загрузка датасета {dataset.name}...")
            train_dataset = load_dataset("json", data_files=dataset.path, split="train")
            print(f"   ✅ Датасет загружен: {len(train_dataset)} примеров")
            
            # Setup LoRA if enabled
            if config["use_lora"]:
                try:
                    from peft import LoraConfig, get_peft_model
                    
                    print("   🔧 Применяем LoRA...")
                    
                    lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM"
                    )
                    
                    loaded_model = get_peft_model(loaded_model, lora_config)
                    loaded_model.print_trainable_parameters()
                    
                except ImportError:
                    print("   ⚠️ PEFT не установлен, LoRA отключен")
                    print("      Установите: pip install peft")
            
            # Training arguments
            print("   ⚙️ Настройка обучения...")
            
            training_args = SFTConfig(
                output_dir=config["output_dir"],
                num_train_epochs=config["epochs"],
                per_device_train_batch_size=config["batch_size"],
                learning_rate=config["lr"],
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                warmup_ratio=0.1,
                gradient_accumulation_steps=4,
                fp16=False,
                bf16=True,
                max_seq_length=2048,
                packing=False,
            )
            
            # Create trainer with our callbacks
            report_path = f"{config['output_dir']}/training_report.md"
            trainer = SFTTrainer(
                model=loaded_model,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                args=training_args,
                callbacks=[
                    ProgressCallback(),
                    EarlyStoppingCallback(patience=3, interactive=True),
                    TrainingReportCallback(output_path=report_path, interactive=False)
                ]
            )
            
            print()
            print("   🚀 НАЧИНАЕМ ОБУЧЕНИЕ!")
            print("=" * 70)
            print()
            
            # Train!
            trainer.train()
            
            print()
            print("=" * 70)
            print("   ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print()
            print(f"   📁 Модель сохранена: {config['output_dir']}")
            print(f"   📄 Отчёт: {config['output_dir']}/training_report.md")
            print("=" * 70)
            
        except ImportError as e:
            print(f"\n   ❌ Отсутствуют зависимости: {e}")
            print("      Установите: pip install trl datasets peft")
            
        except Exception as e:
            print(f"\n   ❌ Ошибка обучения: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n   Нажмите Enter для продолжения...")
    
    def run(self):
        """Main interactive loop."""
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  🔨 TRANSFORMERS FORGE — Interactive Model Manager                   ║")
        print("║  v1.0.9                                                              ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"   📂 Папка моделей:   {self.models_dir.absolute()}")
        print(f"   📁 Папка датасетов: {self.datasets_dir.absolute()}")
        
        while True:
            # Scan models
            models = self.scan_models()
            
            # Select model
            model = self.show_models_menu(models)
            if model is None:
                print("\n   👋 До свидания!")
                break
            
            self.selected_model = model
            
            # Actions loop
            while True:
                action = self.show_actions_menu(model)
                
                if action is None:
                    break
                elif action == "analyze":
                    self.run_analyze(model)
                elif action == "chat":
                    print("\n   💬 Чат будет доступен в следующей версии...")
                    input("   Нажмите Enter...")
                elif action == "finetune":
                    self.run_finetune_wizard(model)


# =============================================================================
# Entry point
# =============================================================================


def interactive_start(models_dir: str = "./models", datasets_dir: str = "./datasets"):
    """
    Start the interactive model manager.
    
    Args:
        models_dir: Path to models directory
        datasets_dir: Path to datasets directory
        
    Example:
        >>> from transformers.interactive import interactive_start
        >>> interactive_start("./models", "./datasets")
    """
    manager = InteractiveModelManager(
        models_dir=models_dir,
        datasets_dir=datasets_dir
    )
    manager.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transformers Forge Interactive Manager")
    parser.add_argument("--models", default="./models", help="Path to models directory")
    parser.add_argument("--datasets", default="./datasets", help="Path to datasets directory")
    
    args = parser.parse_args()
    
    interactive_start(args.models, args.datasets)
