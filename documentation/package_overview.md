# nanochat - Package Documentation & Implementation Plan

## 1. Overview
**nanochat** is a minimalist, full-stack implementation of a Large Language Model (LLM) similar to ChatGPT. It is designed to be hackable, clean, and dependency-light, capable of running on a single 8xH100 node for training or on CPU/MPS for inference.

The project includes:
- **`nanochat`**: A Python package containing the model, training, inference, and serving logic.
- **`rustbpe`**: A high-performance BPE tokenizer implemented in Rust.
- **Scripts**: End-to-end scripts for training (`speedrun.sh`, `run1000.sh`) and performing tasks.

## 2. Dependency Analysis

### 2.1 Python Dependencies (`pyproject.toml`)
The project requires Python >= 3.10.

| Dependency | Purpose |
| :--- | :--- |
| `torch` (>=2.8.0) | Core deep learning framework. Supports CPU and CUDA (12.8). |
| `tiktoken`, `tokenizers` | Tokenization utilities (likely used alongside `rustbpe`). |
| `datasets` | Hugging Face datasets for accessing training data. |
| `fastapi`, `uvicorn` | Web serving for the chat UI. |
| `wandb` | Experiment tracking (Weights & Biases). |
| `psutil`, `regex` | System utilities and regular expressions. |
| `files-to-prompt` | Utility to pack codebase for LLM prompting. |

**Dev Dependencies**: `pytest`, `maturin` (for building the Rust extension).

### 2.2 Rust Dependencies (`rustbpe/Cargo.toml`)
The `rustbpe` crate handles efficient BPE tokenization.

| Dependency | Purpose |
| :--- | :--- |
| `pyo3` | Rust bindings for Python (creates the extension module). |
| `rayon` | Parallelism for faster tokenization. |
| `fancy-regex` | Regex support compatible with Python's regex. |
| `ahash`, `indexmap` | Efficient hashing and map structures. |
| `compact_str` | Memory-efficient string storage. |

## 3. Architecture Breakdown

### 3.1 Core Components (`nanochat/`)
- **`gpt.py`**: The heart of the model. Implements the Transformer architecture with modern features:
    - **Rotary Embeddings (RoPE)**: Replaces positional embeddings.
    - **RMSNorm**: No learnable parameters.
    - **GQA (Group Query Attention)**: Independent key/value heads for efficient inference.
    - **Muon Optimizer**: Specialized optimization supported in `setup_optimizers`.
- **`engine.py`**: High-performance inference engine.
    - **KVCache**: Manages Key-Value cache for autoregressive generation. Supports `prefill` (batch 1) and `decode` phases.
    - **Batch Generation**: Handles multiple samples in parallel.
    - **Tool Use**: Includes a basic `use_calculator` capability via Python `eval`.
- **`tokenizer.py`**: Python wrapper for the BPE tokenizer.
- **`dataloader.py`**: Distributed data loading logic.

### 3.2 Training & Customization
- **Optimizers**: `adamw.py` (Distributed AdamW) and `muon.py` (Distributed Muon).
- **Evaluation**: `core_eval.py` (CORE score), `loss_eval.py` (bits/byte), `chat_eval.py`.
- **Tasks**: `nanochat/tasks/` contains definitions for tasks like ARC, GSM8K, and HumanEval used for evaluation.
- **Scripts**:
    - `speedrun.sh`: Trains a "d20" model in ~4 hours ($100 tier).
    - `run1000.sh`: Trains a "d32" model ($800 tier).

## 4. Key Workflows (Standard)

### 4.1 Inference (Generation)
Inference is handled by `nanochat.engine.Engine`.
1.  **Initialization**: `Engine(model, tokenizer)`
2.  **Generation**: `engine.generate(tokens, ...)`
    -   **Prefill**: Processes the prompt (KV cache populated).
    -   **Decode**: Autoregressively generates token by token.
    -   **Sampling**: Uses temperature and top-k sampling.
    -   **Tools**: Can detect `<|python_start|>` tokens to execute Python code (calculator).

### 4.2 Training Pipeline
Training is orchestrated via shell scripts calling python modules (e.g., `scripts.base_train`).
-   **Distributed**: Uses `torchrun` for multi-GPU support.
-   **Gradient Accumulation**: Automatically adjusts if running on fewer GPUs.
-   **Mixed Precision**: Uses `bfloat16`.

**Standard Stages**:
1.  **Pretraining**: ~3 hours (d20), trains on FineWeb-EDU shards. Scaling law: ~20x params.
2.  **Midtraining**: ~8 mins, adapts to conversation format, special tokens, and multiple choice (MMLU).
3.  **SFT**: ~7 mins, cherry-picked data, individual row padding for domain adaptation.
4.  **RL (Optional)**: GRPO on GSM8K.

---

## 5. User Customization Plan: General English Conversation Model

### 5.1 Project Goal
To build a high-quality "General English Chat Model" without specific focus on coding, math, or deep world knowledge. The model should maintain coherent, natural conversation.

### 5.2 Constraints
-   **Budget**: ~10-12 hours of 8xH100 compute (approx $300-$400 equivalent).
-   **Data**: English-focused (Wikipedia, Books, Reddit).
-   **Task**: Pure conversation.

### 5.3 Custom Data
-   **Tokenizer**: Custom BPE tokenizer already trained.
    -   **Vocabulary**: 64K size.
    -   **Data Split**: 45% Wikipedia, 40% Books, 15% Reddit.
    -   **Performance**: 5.1 average character length.

### 5.4 Proposed Architecture Modifications
To optimize for the conversational goal and budget, the following architectural changes are planned to reduce parameter count and computational cost while preserving conversational capability.

#### A. Weight Tying (Embeddings)
-   **Change**: Tie the weights of the token embedding layer (`wte`) and the language model head (`lm_head`).
-   **Impact**:
    -   Saves `vocab_size * n_embd` parameters (approx. 82M parameters for a d20 model with 64K vocab).
    -   Reduces memory footprint significantly.
-   **Implementation**: modify `gpt.py` to share the weight tensor, update `setup_optimizers` to handle single parameter group.

#### B. Context Length Reduction
-   **Change**: Reduce `max_seq_len` from **2048** to **1024** (or potentially **512**).
-   **Rationale**: General conversation rarely requires massive context windows compared to document analysis or coding.
-   **Impact**:
    -   Reduces memory usage (KV cache).
    -   Reduces Attention mechanism complexity (quadratic scaling).
    -   Allows for larger batch sizes, speeding up training.

#### C. MLP Dimension Reduction
-   **Change**: Reduce the MLP expansion factor from the standard **4x** to **3x** or **2.5x**.
-   **Current**: `n_embd -> 4 * n_embd -> n_embd`
-   **Impact**:
    -   MLP layers constitute ~2/3 of total parameters.
    -   Reducing to 3x saves ~17% of MLP parameters (~100M params).
    -   Trade-off: Slight reduction in expressivity/memorization, acceptable for conversational focus.

#### D. Group-Query Attention (GQA) Tuning
-   **Change**: Set `n_kv_head` to `n_head // 4` (instead of 1:1 or 1:2).
-   **Impact**:
    -   Significantly reduces Key/Value projection parameters.
    -   Lowers inference memory usage (KV Cache).
    -   Standard practice in modern models (Llama 2/3, Gemma).

#### E. Activation Function
-   **Change**: Switch from `relu^2` (squared ReLU) to **GELU** or **SiLU**.
-   **Rationale**: GELU is the standard for modern LLMs (GPT-series, Llama, etc.) and offers smoother gradients. `relu^2` was likely chosen for specific sparsity properties which might not be the priority here.

#### F. Other Optimizations
-   **Logit Softcap**: Evaluate modifying or removing the `softcap=15` applied to logits (Gemma-style regularization).
-   **Rotary Embeddings**: Ensure `base` theta is sufficient (e.g., 100k) if context length remains high, otherwise standard is fine.

### 5.5 Execution Roadmap
1.  **Architecture Update**: Modify `nanochat/gpt.py` to implement the above changes (Weight tying, GQA, MLP ratio).
2.  **Config Update**: Adjust `scripts/base_train.py` and `configurator.py` to support new hyperparameters.
3.  **Training**: Run the training pipeline on the custom tokenizer and English dataset within the allocated 12-hour compute budget.
