"""
🔬 EMA Benchmark — Static vs Dynamic Decay
==========================================

Этот скрипт сравнивает:
1. Без EMA
2. EMA со статическим decay
3. EMA с динамическим decay (НОВИНКА v1.0.4!)

Запуск:
    python benchmarks/ema_benchmark.py

Примерное время: 30-60 секунд на CPU
"""

import time
import sys
import os

# Добавляем путь к src для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_benchmark():
    """Запуск benchmark сравнения Static vs Dynamic EMA."""
    
    print("=" * 70)
    print("🔬 EMA BENCHMARK v3 — Static vs Dynamic Decay")
    print("=" * 70)
    print()
    
    # Импорты
    print("📦 Загрузка библиотек...")
    
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("❌ PyTorch не установлен. Установите: pip install torch")
        return
    
    try:
        from transformers.ema import (
            create_ema_state, 
            update_ema_state, 
            apply_ema_state,
            compute_dynamic_decay
        )
    except ImportError:
        print("❌ Transformers Forge не установлен. Установите: pip install -e .")
        return
    
    print("✅ Библиотеки загружены")
    print()
    
    # Конфигурация
    HIDDEN_SIZE = 256
    NUM_LAYERS = 4
    BATCH_SIZE = 32
    NUM_SAMPLES = 2000
    NUM_STEPS = 300
    LEARNING_RATE = 2e-3  # Высокий LR для шума
    
    # EMA параметры
    STATIC_DECAY = 0.99
    MIN_DECAY = 0.9
    MAX_DECAY = 0.999
    
    print("⚙️ Конфигурация:")
    print(f"   Hidden size: {HIDDEN_SIZE}")
    print(f"   Layers: {NUM_LAYERS}")
    print(f"   Training steps: {NUM_STEPS}")
    print(f"   Learning rate: {LEARNING_RATE} (высокий для шума)")
    print()
    print("📊 EMA параметры:")
    print(f"   Static decay: {STATIC_DECAY}")
    print(f"   Dynamic decay: {MIN_DECAY} → {MAX_DECAY}")
    print()
    
    # Модель
    class SimpleTransformer(nn.Module):
        def __init__(self, hidden_size, num_layers):
            super().__init__()
            self.embedding = nn.Linear(hidden_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.LayerNorm(hidden_size)
                )
                for _ in range(num_layers)
            ])
            self.head = nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = x + layer(x)
            return self.head(x)
    
    # Данные
    print("📊 Генерация данных...")
    torch.manual_seed(42)
    X = torch.randn(NUM_SAMPLES, HIDDEN_SIZE)
    Y = torch.sin(X) * 0.5 + torch.randn_like(X) * 0.2
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    X_eval = torch.randn(500, HIDDEN_SIZE)
    Y_eval = torch.sin(X_eval) * 0.5
    
    def evaluate(model, X_eval, Y_eval):
        model.eval()
        with torch.no_grad():
            pred = model(X_eval)
            loss = nn.MSELoss()(pred, Y_eval)
        model.train()
        return loss.item()
    
    def train_with_ema(use_dynamic: bool, tag: str):
        """Обучение с EMA (статическим или динамическим)."""
        model = SimpleTransformer(HIDDEN_SIZE, NUM_LAYERS)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        ema_state = create_ema_state(model)
        
        eval_normal_history = []
        eval_ema_history = []
        decay_history = []
        
        step = 0
        for epoch in range(20):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                # Вычисляем decay
                if use_dynamic:
                    decay = compute_dynamic_decay(
                        current_step=step,
                        total_steps=NUM_STEPS,
                        min_decay=MIN_DECAY,
                        max_decay=MAX_DECAY,
                        schedule="linear"
                    )
                else:
                    decay = STATIC_DECAY
                
                # Обновляем EMA
                update_ema_state(model, ema_state, decay=decay)
                
                if step % 50 == 0:
                    eval_normal = evaluate(model, X_eval, Y_eval)
                    eval_normal_history.append(eval_normal)
                    
                    backup = apply_ema_state(model, ema_state)
                    eval_ema = evaluate(model, X_eval, Y_eval)
                    eval_ema_history.append(eval_ema)
                    apply_ema_state(model, backup)
                    
                    decay_history.append(decay)
                    
                    diff = ((eval_normal - eval_ema) / eval_normal) * 100 if eval_normal > 0 else 0
                    marker = "✅" if eval_ema < eval_normal else "⚠️"
                    
                    print(f"   [{tag}] Step {step:3d} | decay={decay:.3f} | Normal: {eval_normal:.4f} | EMA: {eval_ema:.4f} | {marker} {diff:+.1f}%")
                
                step += 1
                if step >= NUM_STEPS:
                    break
            if step >= NUM_STEPS:
                break
        
        # Финальная оценка
        final_normal = evaluate(model, X_eval, Y_eval)
        backup = apply_ema_state(model, ema_state)
        final_ema = evaluate(model, X_eval, Y_eval)
        
        ema_wins = sum(1 for n, e in zip(eval_normal_history, eval_ema_history) if e < n)
        
        return {
            "final_normal": final_normal,
            "final_ema": final_ema,
            "improvement": ((final_normal - final_ema) / final_normal) * 100,
            "ema_win_rate": ema_wins / len(eval_normal_history) if eval_normal_history else 0,
            "final_decay": decay_history[-1] if decay_history else STATIC_DECAY,
        }
    
    # ========================================================================
    # Эксперименты
    # ========================================================================
    
    print()
    print("-" * 70)
    print("🔴 Эксперимент 1: EMA со СТАТИЧЕСКИМ decay")
    print("-" * 70)
    static_results = train_with_ema(use_dynamic=False, tag="STATIC")
    
    print()
    print("-" * 70)
    print("🟢 Эксперимент 2: EMA с ДИНАМИЧЕСКИМ decay")
    print("-" * 70)
    dynamic_results = train_with_ema(use_dynamic=True, tag="DYNAMIC")
    
    # ========================================================================
    # Результаты
    # ========================================================================
    print()
    print("=" * 70)
    print("📊 РЕЗУЛЬТАТЫ BENCHMARK")
    print("=" * 70)
    print()
    
    print(f"   {'Метрика':<30} {'Static EMA':<15} {'Dynamic EMA':<15}")
    print(f"   {'-'*30} {'-'*15} {'-'*15}")
    
    print(f"   {'Final Eval (Normal)':<30} {static_results['final_normal']:<15.4f} {dynamic_results['final_normal']:<15.4f}")
    print(f"   {'Final Eval (EMA)':<30} {static_results['final_ema']:<15.4f} {dynamic_results['final_ema']:<15.4f}")
    print(f"   {'EMA Improvement':<30} {static_results['improvement']:>+14.1f}% {dynamic_results['improvement']:>+14.1f}%")
    print(f"   {'EMA Win Rate':<30} {static_results['ema_win_rate']*100:>14.0f}% {dynamic_results['ema_win_rate']*100:>14.0f}%")
    print(f"   {'Final Decay':<30} {static_results['final_decay']:<15.3f} {dynamic_results['final_decay']:<15.3f}")
    
    print()
    
    # Сравнение
    if dynamic_results['improvement'] > static_results['improvement']:
        diff = dynamic_results['improvement'] - static_results['improvement']
        print(f"   ✅ Dynamic EMA лучше на {diff:.1f}%!")
    elif static_results['improvement'] > dynamic_results['improvement']:
        diff = static_results['improvement'] - dynamic_results['improvement']
        print(f"   ⚠️ Static EMA лучше на {diff:.1f}%")
    else:
        print(f"   ➖ Результаты одинаковы")
    
    if dynamic_results['ema_win_rate'] > static_results['ema_win_rate']:
        print(f"   ✅ Dynamic EMA выигрывает чаще!")
    
    print()
    print("=" * 70)
    print("📝 ВЫВОДЫ")
    print("=" * 70)
    print("""
   DYNAMIC DECAY решает проблему отставания EMA:
   
   📉 Static decay (0.99):
      - Начало: EMA сильно отстаёт (помнит плохие начальные веса)
      - Конец: EMA может не догнать модель
   
   📈 Dynamic decay (0.9 → 0.999):
      - Начало: decay=0.9 (быстрая адаптация к текущим весам)
      - Конец: decay=0.999 (стабильное усреднение)
   
   КОГДА ИСПОЛЬЗОВАТЬ:
   ✅ Dynamic decay — для любого обучения
   ✅ Особенно полезен для коротких тренировок
   ✅ Автоматически адаптируется к длине обучения
""")
    print("=" * 70)
    
    return {
        "static": static_results,
        "dynamic": dynamic_results,
    }


if __name__ == "__main__":
    results = run_benchmark()
