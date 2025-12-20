# nanochat - Package Documentation

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

## 4. Key Workflows

### 4.1 Inference (Generation)
Inference is handled by `nanochat.engine.Engine`.
1.  **Initialization**: `Engine(model, tokenizer)`
2.  **Generation**: `engine.generate(tokens, ...)`
    -   **Prefill**: Processes the prompt (KV cache populated).
    -   **Decode**: Autoregressively generates token by token.
    -   **Sampling**: Uses temperature and top-k sampling.
    -   **Tools**: Can detect `<|python_start|>` tokens to execute Python code (calculator).

### 4.2 Training
Training is orchestrated via shell scripts calling python modules (e.g., `scripts.base_train`).
-   **Distributed**: Uses `torchrun` for multi-GPU support.
-   **Gradient Accumulation**: Automatically adjusts if running on fewer GPUs.
-   **Mixed Precision**: Uses `bfloat16`.

## 5. Directory Structure Summary
```
nanochat_eng/
├── nanochat/          # Main Python package
│   ├── gpt.py         # Model definition
│   ├── engine.py      # Inference engine
│   └── ...
├── rustbpe/           # Rust tokenizer extension
├── scripts/           # Training and evaluation scripts
├── tasks/             # Task definitions for evaluation
├── dev/               # Development utilities
├── tests/             # Tests (pytest)
├── pyproject.toml     # Python config
└── Cargo.toml         # Rust config (inside rustbpe)
```

## 6. Deep Dive: Implementation Details

### 6.1 Data Pipeline
The data pipeline is designed for massive scale and distributed training.
-   **Storage**: Data is stored as Parquet files (shards) in `base_data/`.
-   **Downloading**: `nanochat.dataset.download_single_file` fetches shards from HuggingFace on demand.
-   **Loading**: `nanochat.dataloader.tokenizing_distributed_data_loader_with_state` is the core generator.
    -   **Distributed**: Each DDP rank reads a subset of row groups from the Parquet files to avoid overlap.
    -   **Tokenization**: Happens on-the-fly using the tokenizer.
    -   **Resumption**: It yields a `state_dict` containing (`pq_idx`, `rg_idx`) to allow approximate resumption of training without re-consuming seen data.
    -   **Performance**: Uses `pin_memory` and `non_blocking` transfers to GPU.

### 6.2 Custom Optimizers (Muon)
`nanochat` uses a hybrid optimization strategy defined in `gpt.GPT.setup_optimizers`:
-   **AdamW**: Used for embeddings and the final language model head.
-   **Muon**: Used for internal 2D linear layers (matrix parameters).
    -   **Concept**: MomentUm Orthogonalized by Newton-schulz. Optimizes matrices by maintaining their spectral norm ~1.
    -   **Implementation**: `nanochat.muon.DistMuon` handles its own distributed synchronization. It uses `reduce_scatter` to average gradients and `all_gather` to sync updated weights, optimizing communication for these large matrices.
    -   **Newton-Schulz**: Uses a quintic iteration to strictly orthogonalize updates.

### 6.3 Tokenizer Implementation
The project uses a dual-backend approach for tokenization:
-   **Training (`rustbpe`)**: A custom Rust extension (`rustbpe.Tokenizer`) handles high-performance BPE training.
    -   **Parallelism**: Uses `rayon` for parallel text processing.
    -   **Algorithm**: Standard BPE with GPT-4's split pattern.
-   **Inference (`tiktoken`)**: A Python wrapper `nanochat.tokenizer.RustBPETokenizer` loads the trained vocabulary into `tiktoken` for extremely fast inference.
-   **Chat Formatting**: `render_conversation` handles converting chat messages (User/Assistant) into token IDs with appropriate masking (training only on Assistant outputs).
