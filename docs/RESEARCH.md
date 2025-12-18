# 📚 Теория и Практика — Transformers Forge

Научное обоснование технологий используемых в Transformers Forge.

---

## 🔬 EMA (Exponential Moving Average)

### 📖 Теория

**Математическое обоснование:**

EMA основан на работе **Polyak & Juditsky (1992)** "Acceleration of Stochastic Approximation by Averaging".

**Формула:**
```
θ_ema(t) = β × θ_ema(t-1) + (1-β) × θ(t)
```

**Ключевой результат:**
- Стандартный SGD сходится как O(1/√t)
- SGD с Polyak averaging сходится как **O(1/t)** — быстрее!

**Почему работает:**
1. SGD "прыгает" вокруг оптимума из-за шума градиентов
2. EMA усредняет эти колебания
3. Результат: меньшая дисперсия → лучшая generalization

**Теоретический выигрыш:**
| Метрика | Без EMA | С EMA | Источник |
|---------|---------|-------|----------|
| Variance | O(σ²/t) | O(σ²/t²) | Polyak 1992 |
| Convergence | O(1/√t) | O(1/t) | Ruppert 1988 |

**Ключевые публикации:**
- Polyak, B. T., & Juditsky, A. B. (1992). "Acceleration of stochastic approximation by averaging"
- Ruppert, D. (1988). "Efficient estimations from a slowly convergent Robbins-Monro process"
- Mandt, S., et al. (2017). "Stochastic Gradient Descent as Approximate Bayesian Inference"

**SOTA модели использующие EMA:**
- ✅ Stable Diffusion (Stability AI)
- ✅ DALL-E 2 (OpenAI)
- ✅ Imagen (Google Brain)
- ✅ EDM (Karras et al., 2022)

---

### 🧪 Практика

**Ожидаемое улучшение:** +1-3% на eval метриках

| Статус | Описание |
|--------|----------|
| 🔴 **Предстоит** | Собственные benchmarks на LLM fine-tuning |

**Планируемые эксперименты:**
- [ ] GPT-2 fine-tuning: loss comparison
- [ ] Mistral-7B QLoRA: perplexity comparison
- [ ] Ablation study: decay values (0.99, 0.999, 0.9999)

---

## 🧊 Layer Freezing (LP-LoRA стиль)

### 📖 Теория

**Концепция:**

Нижние слои transformer содержат более общие представления (syntax, basic semantics), а верхние — task-specific знания.

**Teоретическое обоснование:**
- **Clark et al. (2019)** "What Does BERT Look At?" — анализ attention patterns
- **Kovaleva et al. (2019)** "Revealing the Dark Secrets of BERT"
- **Guo et al. (2023)** "LongLoRA" — эффективное расширение контекста с freezing

**Почему работает:**
1. Нижние слои уже хорошо обучены на pretrain
2. Заморозка предотвращает "catastrophic forgetting"
3. Меньше параметров = меньше памяти на градиенты

**LP-LoRA (Layer-wise Partial LoRA):**
- Freeze 50% нижних слоёв
- LoRA только на верхних слоях
- Результат: экономия 50% памяти на градиенты

**Теоретическая экономия памяти:**
| % заморозки | Экономия градиентов | Экономия optimizer |
|-------------|---------------------|---------------------|
| 25% | 25% | 25% |
| 50% | 50% | 50% |
| 75% | 75% | 75% |

---

### 🧪 Практика

| Статус | Описание |
|--------|----------|
| 🔴 **Предстоит** | Benchmarks memory usage и quality |

**Планируемые эксперименты:**
- [ ] Memory profiling: 0% vs 50% vs 75% frozen
- [ ] Quality comparison: frozen vs full fine-tuning
- [ ] Optimal freeze ratio для разных задач

---

## ⚙️ Training Presets

### 📖 Теория

**Концепция:**

Готовые конфигурации основаны на best practices из:
- Hugging Face PEFT documentation
- QLoRA paper (Dettmers et al., 2023)
- TRL library defaults

**SFT (Supervised Fine-Tuning):**
- Learning rate: 2e-5 (стандарт для BERT-style моделей)
- Warmup: 10% (предотвращает destabilization в начале)
- Cosine schedule (постепенное снижение lr)

**LoRA parameters:**
- r=16: баланс качество/эффективность (Hu et al., 2021)
- lora_alpha=32: рекомендуемый alpha=2*r
- target_modules: q,k,v,o projections (максимальный эффект)

**QLoRA:**
- NF4 quantization: лучше чем INT4 для weights (Dettmers 2023)
- Double quantization: дополнительная экономия памяти
- BF16 compute: лучшая точность чем FP16

---

### 🧪 Практика

| Статус | Описание |
|--------|----------|
| 🔴 **Предстоит** | Сравнение presets на стандартных бенчмарках |

**Планируемые эксперименты:**
- [ ] SFT vs LoRA vs QLoRA: quality/speed tradeoff
- [ ] Preset defaults vs custom tuning
- [ ] Memory usage comparison

---

## 📊 Training Monitor

### 📖 Теория

**Концепция:**

Мониторинг обучения для раннего обнаружения проблем:

**Gradient Health:**
- **Vanishing gradients**: norm < 1e-7 (модель не учится)
- **Exploding gradients**: norm > 1e3 (нестабильность)
- **NaN/Inf**: критическая ошибка

**Memory Estimation:**
- Parameters: 4 bytes × num_params (FP32) или 2 bytes (FP16)
- Gradients: такой же размер как parameters (для trainable)
- Optimizer: 2x gradients для Adam/AdamW (momentum + variance)

---

### 🧪 Практика

| Статус | Описание |
|--------|----------|
| ✅ **Проверено** | Unit тесты на gradient detection |
| 🔴 **Предстоит** | Real-world monitoring во время обучения |

---

## 📈 Общий статус

| Модуль | Теория | Практика | Тесты |
|--------|--------|----------|-------|
| EMA | ✅ Доказано | 🔴 Предстоит | ✅ Работают |
| Layer Freezing | ✅ Обосновано | 🔴 Предстоит | ✅ Работают |
| Training Presets | ✅ Best practices | 🔴 Предстоит | ✅ Работают |
| Training Monitor | ✅ Стандартные метрики | 🔴 Предстоит | ✅ Работают |

---

## 📚 Библиография

1. Polyak, B. T., & Juditsky, A. B. (1992). "Acceleration of stochastic approximation by averaging". SIAM Journal on Control and Optimization.

2. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models". arXiv:2106.09685.

3. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs". arXiv:2305.14314.

4. Karras, T., et al. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models". NeurIPS.

5. Clark, K., et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention". BlackboxNLP.

---

**Последнее обновление:** Декабрь 2025
