# Leaking LoRA

This repository contains the code and notebook used to study memorization and recall of sensitive strings in large language models using LoRA fine-tuning and ROME editing.

**Important structure:**
- The **notebook** contains the core research workflow and experiments.
- The **CLI script** exists to make the experiments reproducible from the command line.

The CLI simply wraps the notebook logic into a reproducible pipeline.

---

# Research Overview

This project investigates:

- Whether LoRA fine‑tuning induces memorization of injected secrets (e.g., passwords)
- Whether memorized strings can be extracted via generation
- Whether ROME editing can suppress recall
- The performance trade‑off after editing

Pipeline:

1. Benchmark base model  
2. Inject password‑like strings into training data  
3. LoRA fine‑tune  
4. Measure recall  
5. Apply ROME editing  
6. Measure recall + performance again  

---

# Notebook vs CLI

## Notebook
The notebook contains:
- Full experimental logic
- Visual inspection of results
- Iterative research workflow
- Used for generating research findings

## CLI script
The CLI version exists purely for:

- Reproducibility
- Running experiments headless
- Batch experiments
- Easier replication 


---

# Installation

```bash
pip install torch transformers peft pandas numpy requests
```

Optional but recommended:

```bash
pip install accelerate
```

---

# Basic Usage

```bash
python main.py
```

Runs ipeline with default settings.

---

#  CLI Parameters

The CLI can be extended with parameters for reproducibility.

Example:

```bash
python main.py   --model facebook/opt-1.3b   --num_passwords 200   --epochs 50   --batch_size 4   --lr 1e-4   --rome_alpha 0.01   --tests 200
```

## Parameter explanations

### --model
Base HuggingFace model used.

Example:
```
--model facebook/opt-1.3b
--model gpt2
```

---

### --num_passwords
Number of RockYou passwords injected into training.

Controls memorization difficulty.

Example:
```
--num_passwords 200
```

---

### --epochs
Number of LoRA training epochs.

Higher values → stronger memorization.

Example:
```
--epochs 50
```

---

### --batch_size
Training batch size.

Example:
```
--batch_size 4
```

---

### --lr
Learning rate for LoRA fine‑tuning.

Example:
```
--lr 1e-4
```

---

### --tests
Number of generation tests used to attempt recall.

Higher = more thorough extraction attempt.

Example:
```
--tests 200
```

---

### --rome_alpha
Strength of ROME edit.

Lower = subtle edit  
Higher = stronger suppression but more risk to model performance.

Example:
```
--rome_alpha 0.01
```

---

# Hardware Requirements

Recommended:
- GPU with ≥16GB VRAM
- CUDA environment

Possible but slow:
- CPU only

---

#  Outputs

The script prints:

- Initial benchmark accuracy
- Post‑LoRA accuracy
- Number of recalled passwords
- Recall after ROME
- Final benchmark accuracy

---

# Ethics & Intended Use

This repository exists for:

- Studying memorization in LLMs
- Evaluating leakage risks
- Testing mitigation strategies

Not intended for:
- Real credential extraction
- Offensive use

Use responsibly and only for research.

---

# Reproducibility Notes

To reproduce paper results:

1. Use same base model
2. Same password count
3. Same LoRA config
4. Same seed (if added)
5. Run full pipeline

Notebook contains original experimental environment.


