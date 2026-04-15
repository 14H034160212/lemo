---
library_name: transformers
tags:
- reasoning
- logic
- logic-inertia
- qwen2
- dpo
- multi-step-inference
license: apache-2.0
datasets:
- qbao775/structural-robustness-benchmark
metrics:
- accuracy
- SR_macro
---

# Fusion-Conflict-Qwen3-8B: Mitigating Logic Inertia in LLMs

This model is a specialized version of **Qwen3-8B** trained to mitigate **Logic Inertia**—the tendency of language models to persist in deductive reasoning even when premises are contradictory or invalid.

It implements the **Fusion-Conflict** framework, which enforces an explicit structural separation between premise verification and deductive execution.

## 📖 Key Information
- **Paper**: [Conflict-Aware Fusion: Mitigating Logic Inertia in Large Language Models](https://arxiv.org/abs/2512.06393)
- **Repository**: [Lemo Project](https://github.com/qbao775/lemo)
- **Goal**: Ensure 100% structural robustness (SR_macro = 1.0) under contradictory premises while maintaining state-of-the-art general logic performance.

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "qbao775/Fusion-Conflict-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Example with contradictory premises
prompt = """Facts:
1. Sensor A reports high temperature.
2. Satellite imagery shows no fire.
3. High temperature implies fire.

Question: Is there a fire?
Answer:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🧠 Training Methodology
The model follows a rigorous four-stage optimization pipeline:
1. **Stage 1: Conflict-Aware SFT**: Finetuning with a "verification-before-deduction" structural prior.
2. **Stage 2: Structural DPO**: Alignment using ~11,200 preference pairs that penalize "Logic Inertia".
3. **Stage 3: Logical Invariance REgularization (LIRE)**: Adding a KL-divergence regularization that forces the model to produce consistent predictions across logically equivalent perturbations (e.g., De Morgan's Law vs. Standard Form).
4. **Stage 4: Reinforcement Learning from Verification Feedback (RLVF)**: Final policy optimization using environmental verification signals to reward conflict detection and deductive accuracy.

## 🌋 Application: Disaster Prediction & multi-source Reasoning
In high-stakes scenarios like disaster management, models often receive information from diverse and potentially conflicting sources (e.g., IoT sensors, citizen reports, satellite data). 

**Fusion-Conflict-8B** acts as a "logic circuit breaker":
- **Conflict Resolution**: Instead of hallucinating a resolution or blindly following one rule, the model identifies the structural contradiction and halts the derivation of unsafe conclusions.
- **Robust Integration**: It can act as a reasoning backbone for disaster-response systems, ensuring that contradictory inputs don't lead to false-positive alarms or missed warnings due to deductive momentum.

### Future Work & Ideas
- **Domain-Specific Logic**: Fine-tuning on disaster-specific NLI datasets to handle domain vocabulary (e.g., specialized meteorological terms).
- **Probalistic Fusion**: Extending the binary conflict-aware prior to handle probabilistic or weighted evidence conflicts.

## 📝 Citation
If you use this model or the framework, please cite:
```bibtex
@article{bao2026fusion,
  title={Conflict-Aware Fusion: Mitigating Logic Inertia in Large Language Models via Structured Cognitive Priors},
  author={Bao, Qiming and Fu, Xiaoxuan and Witbrock, Michael},
  journal={arXiv preprint arXiv:2512.06393},
  year={2026}
}
```
