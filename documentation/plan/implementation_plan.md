## Implementation Plan

### Tokenizer

- If same dataset is not used for training LLM, it can increase risk of unknown characters in the training dataset.
    - Validate if new tokenizer can be used for default dataset also or some other dataset can be used for training that is english characters dominant
    - Otherwise add all characters from training dataset to tokenizer
- Target 5+ compression rate otherwise go with default tokenizer
- Plan - Training a  english-dominant model so tokenizer will be trained on a different dataset as fineweb-edu is code heavy, has other languages and has math symbols.
- New tokenizer dataset -
    - Options 1 - Wikipedia
    - Options 2 - 50% Wikipedia + 30% Books Corpus + 20% news or conversational text
- Datasets - not curating new tokenizer dataset. It is a solved problem with many freely available datasets.
- Vocab size - 32K


#### Tokenizer training results
- Option 2 had better performance
- Datasets used - 
  - 45% Wikipedia (Google/Wiki40b)
  - 40% Books Corpus (Navanjana/Gutenberg_books)
  - 15% reddit comments (sentence-transformers/reddit)

- Trained 2 tokenizers - 32K and 64K vocab
- 64K beats GPT-4 on all relevant datasets
- 32K is close to 64K on reddit and wiki datasets but lags generic in english

```
Vocab sizes:
GPT-2: 50257
GPT-4: 100277
Ours: 65536

Comparison with GPT-2:
===============================================================================================
Text Type  Bytes    GPT-2           Ours            Relative     Better    
                    Tokens  Ratio   Tokens  Ratio   Diff %      
-----------------------------------------------------------------------------------------------
english    2681     551     4.87    522     5.14       +5.3%     Ours      
korean     893      745     1.20    775     1.15       -4.0%     GPT-2     
code       1259     576     2.19    410     3.07      +28.8%     Ours      
math       1834     936     1.96    1017    1.80       -8.7%     GPT-2     
science    1112     260     4.28    243     4.58       +6.5%     Ours      
gutenberg  100002   21792   4.59    21357   4.68       +2.0%     Ours      
wiki       100174   21678   4.62    21574   4.64       +0.5%     Ours      
reddit     100142   24226   4.13    23951   4.18       +1.1%     Ours      

Comparison with GPT-4:
===============================================================================================
Text Type  Bytes    GPT-4           Ours            Relative     Better    
                    Tokens  Ratio   Tokens  Ratio   Diff %      
-----------------------------------------------------------------------------------------------
english    2681     529     5.07    522     5.14       +1.3%     Ours      
korean     893      364     2.45    775     1.15     -112.9%     GPT-4     
code       1259     309     4.07    410     3.07      -32.7%     GPT-4     
math       1834     832     2.20    1017    1.80      -22.2%     GPT-4     
science    1112     249     4.47    243     4.58       +2.4%     Ours      
gutenberg  100002   21396   4.67    21357   4.68       +0.2%     Ours      
wiki       100174   22023   4.55    21574   4.64       +2.0%     Ours      
reddit     100142   23149   4.33    23951   4.18       -3.5%     GPT-4 
```


```
Vocab sizes:
GPT-2: 50257
GPT-4: 100277
Ours: 32768

Comparison with GPT-2:
===============================================================================================
Text Type  Bytes    GPT-2           Ours            Relative     Better    
                    Tokens  Ratio   Tokens  Ratio   Diff %      
-----------------------------------------------------------------------------------------------
english    2681     551     4.87    570     4.70       -3.4%     GPT-2     
korean     893      745     1.20    853     1.05      -14.5%     GPT-2     
code       1259     576     2.19    429     2.93      +25.5%     Ours      
math       1834     936     1.96    1069    1.72      -14.2%     GPT-2     
science    1112     260     4.28    275     4.04       -5.8%     GPT-2     
gutenberg  100002   21792   4.59    22157   4.51       -1.7%     GPT-2     
wiki       100174   21678   4.62    22512   4.45       -3.8%     GPT-2     
reddit     100142   24226   4.13    24703   4.05       -2.0%     GPT-2     

Comparison with GPT-4:
===============================================================================================
Text Type  Bytes    GPT-4           Ours            Relative     Better    
                    Tokens  Ratio   Tokens  Ratio   Diff %      
-----------------------------------------------------------------------------------------------
english    2681     529     5.07    570     4.70       -7.8%     GPT-4     
korean     893      364     2.45    853     1.05     -134.3%     GPT-4     
code       1259     309     4.07    429     2.93      -38.8%     GPT-4     
math       1834     832     2.20    1069    1.72      -28.5%     GPT-4     
science    1112     249     4.47    275     4.04      -10.4%     GPT-4     
gutenberg  100002   21396   4.67    22157   4.51       -3.6%     GPT-4     
wiki       100174   22023   4.55    22512   4.45       -2.2%     GPT-4     
reddit     100142   23149   4.33    24703   4.05       -6.7%     GPT-4 
```


### Pretraining

#### Model Architecture

- Vocab size - ?
- Head size - 128
- Number of heads - ?
- Width (embedding size) - ?
- Depth - ?
- max_seq_len (Context window) - 2048 


#### Modification Options
- depth vs number of heads tradeoff for conversation vs world knowledge
  - Saves vocab_size × n_embd parameters (~82M for 64K × 1280)
  - In gpt.py: Use self.lm_head.weight = self.transformer.wte.weight
  - Optimizer impact: The tied weights need a single optimizer group (likely AdamW, not separate embedding_lr/unembedding_lr)
- MLP hidden dimension size tradeoff. Can it be reduced to 3 × n_embd or even 2.5 × n_embd
  - Currently: 4 × n_embd expansion (line 115-116)
    - self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
  - Option: Reduce to 3 × n_embd or even 2.5 × n_embd
    - Saves ~25-40% of MLP parameters (MLP is ~2/3 of model params)
    - Trade-off: Reduced expressivity, but for conversation vs. world knowledge, this may be acceptable
- Reduce Context Length (max_seq_len)
  - Currently: 2048 tokens
  - For conversational focus, you likely don't need long context:
    - 1024 saves 50% attention memory → allows larger batches
    - 512 for very short exchanges (but may hurt coherence)
  - Impact: Reduces rotary_seq_len and attention FLOPs quadratically
- Activation Function
  - Currently uses relu^2 (squared ReLU):
    - x = F.relu(x).square()  # line 120
    - Alternative: Standard GELU or SiLU/Swish
  - relu^2 is good for sparsity but may not be critical for conversation
  - GELU is more common and well-understood
- Group-Query Attention (GQA)
  - Currently: n_kv_head = n_head (1:1, GQA disabled)
  - Option: Set n_kv_head = n_head // 4 or n_head // 2
    - Saves ~50-75% of K/V projection parameters per layer
    - Also saves KV cache memory during inference
    - Popular in Llama 2/3, Gemma