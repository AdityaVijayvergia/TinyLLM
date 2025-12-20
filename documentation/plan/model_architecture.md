# Nanochat Model Architecture Explained

This document outlines the architecture of the `nanochat` model, a minimal yet state-of-the-art Large Language Model (LLM) implementation. It uses a logic where most dimensions are derived from a single "Depth" parameter, making it an excellent educational example of Transformer scaling.

## 1. The Scaling Heuristic (The "Golden Rule")

In this architecture, you don't pick the width or number of heads manually. You pick how "deep" you want the model to be, and the rest is calculated to maintain a balanced "Aspect Ratio".

*   **User Input**: `depth` (e.g., 20 layers).
*   **Calculated Width (`n_embd`)**: `depth * 64`
    *   *Rationale*: As models get deeper, they generally need to get wider to propagate gradients effectively. A ratio of ~64 is a common rule of thumb.
    *   *Example*: `20 * 64 = 1280` dimensions.
*   **Calculated Heads**: `width / 128`
    *   *Rationale*: Each attention head processes a *subspace* of the data.
        *   **Why 128?**: This is a standard size chosen for GPU efficiency (powers of 2 like 64, 128 work best). It's a "sweet spot": large enough to encode complex relationships/features, but small enough to be computed quickly.
        *   **"Process a Subspace"**: Imagine the full 1280-dimension vector represents the entire "meaning" of a token. Splitting it into 10 heads of size 128 is like assigning 10 different specialists. One head might focus only on grammar (subject-verb agreement), another on long-term context (referencing an earlier paragraph), and another on translation. Each head processes only its slice (subspace) of the total information.
    *   *Example*: `1280 / 128 = 10` heads.

## 2. The Architecture Flowchart

This is the journey of data through the "brain" of the model.

### Stage 1: The Entryway (Embeddings)
**Input**: A list of integers (Token IDs). Shape: `(Batch_Size, Sequence_Length)`

*   **The Embedding Matrix (`wte`)**: A giant lookup table.
    *   **Size**: `Vocab_Size x Width` (~50,000 x 1,280).
    *   **Action**: Each discrete token ID (e.g., `412`) looks up its corresponding continuous vector (size 1,280).

### Stage 2: The Processing Stack (The "Depth")
The data enters a loop of **N Layers**. Inside *each* Layer (`Block`), the vector representation is refined:

1.  **RMSNorm**: The vector is normalized (scaled) to stabilize training (Pre-Norm architecture).
2.  **Attention ("Communication")**:
    *   **Concept**: This is the *only* time tokens share information with each other. The model looks at previous tokens to understand context.
    *   **Heads**: The vector is split into smaller vectors (Heads). Each head learns different relationships (grammar, semantic meaning, etc.).
    *   **RoPE (Rotary Positional Embeddings)**: Instead of fixed position numbers, vectors are mathematically "rotated" to encode relative distances between tokens.
3.  **Residual Connection**: The result of Attention is *added* back to the original vector (`x = x + Attention(x)`).
    *   **Constant Shape**: This requires the input `x` and output `Attention(x)` to have the exact same shape `(Batch, Sequence, Width)`. This shape remains constant throughout the entire model (until the very end).
    *   **The "Gradient Highway"**: This is crucial. During training, gradients need to flow backward from the last layer to the first. In a standard chain (`layer(layer(...))`), gradients multiply and can vanish (become zero) or explode (become infinity) after many layers.
    *   **Why add?**: With addition, the gradient can flow *directly* through the `+` operation unmodified, "skipping" layers if needed. This allows us to train very deep networks (100+ layers) without the signal degrading. It implies the model learns "refinements" to the vector rather than rewriting it completely at each step.
4.  **MLP ("Thinking")**:
    *   **Structure**: `Width` -> `4 * Width` -> `Width`
    *   **Action**: The vector is expanded to 4x its size, processed with a non-linearity (`ReLU` squared), and projected back down.
    *   **Concept**: This processes each token *individually* to extract higher-level features from the information gathered by Attention.

### Stage 3: The Exit (The Logits)
After passing through all layers:

1.  **RMSNorm**: Final normalization.
2.  **Linear Head (`lm_head`)**: The inverse of the embedding.
    *   **Action**: Projects the vector back up to the vocabulary size (e.g., 1,280 -> 50,000).
    *   **Output (Logits)**: A score for every possible word in the dictionary. The highest score represents the predicted next word.

## 3. Summary of Dimensions (Example "d20" Config)

| Concept | Variable | Example Value | Meaning |
| :--- | :--- | :--- | :--- |
| **Depth** | `n_layer` | **20** | How many reasoning steps the model can take. |
| **Width** | `n_embd` | **1280** | Capacity of "information" a single word's state can hold. |
| **Heads** | `n_head` | **10** | Number of independent "views" of the context. (See below for impact) |
| **Context** | `sequence_len` | **2048** | How many previous words it can remember at once. |
| **Vocab** | `vocab_size` | **~50k** | The dictionary size (e.g., GPT-4's cl100k). |

### Impact of Changing the Number of Heads
The number of heads controls the model's ability to "multitask" its attention.
*   **More Heads (e.g., 20 heads of size 64)**: The model can track more distinct relationships at once (e.g., who is speaking, what the subject is, the tone, the tense). However, each "view" is lower resolution (fewer dimensions to represent that relationship).
*   **Fewer Heads (e.g., 5 heads of size 256)**: Each head has a very high-fidelity view of the data and can model complex interactions. However, the model can focus on fewer distinct things simultaneously.
*   **The Trade-off**: There is an empirical "Goldilocks zone." Too few heads, and the model misses connections. Too many, and it gets confused by noisy, low-resolution signals. The `Width / 128` heuristic is the current industry standard balance.


This configuration results in a model with roughly **400-500 Million parameters**, designed to be trainable on consumer hardware ("nano" scale).

## 4. Architectural Decisions & Notes

### Decision: Finalized Configuration (d18 + Tied Weights)
**Date**: 2025-12-10
**Goal**: Balance reasoning capability with training cost (Knowledge is not a priority).

*   **Configuration**: `depth = 18`
    *   **Implication**: `Width = 18 * 64 = 1152`, `Heads = 9`.
*   **Optimization**: **Tied Embeddings**
    *   `lm_head.weight = wte.weight`.
    *   **Savings**: `50,304 (Vocab) * 1,152 (Width) ≈ 58 Million Parameters`.
*   **Resulting Model**:
    *   **Estimated Size**: **~300 Million Parameters**.
    *   **Trade-off**:
        *   Significantly cheaper/faster than the baseline d20 or d24.
        *   More capable than the budget d16.
        *   Knowledge will be weak (expected), but conversation flow and basic reasoning should remain intact.

## 5. Proposed Experiments (Pending)

### Experiment: Custom "English-Heavy" Tokenizer
**Status**: Under Consideration
**Hypothesis**: Reducing vocabulary size from ~50k to ~32k (removing code/multilingual tokens) could save parameters.

*   **Potential Gains**:
    *   Reducing vocab by ~20k tokens saves ~25M params (if untied) or reduces the embedding table size.
    *   Could allow for slightly more depth (~1 extra layer).
*   **Risks**:
    *   **Sequence Inflation**: Smaller vocabularies break words into more pieces.
    *   **Context Loss**: "The same amount of text takes up more context window space."
    *   **Compute Cost**: The model has to process more individual tokens to understand the same sentence.
*   **Current Verdict**: Low priority. The 5% parameter savings might not be worth the risk of inefficient tokenization. Will review BPE (Byte Pair Encoding) logic first before deciding.
