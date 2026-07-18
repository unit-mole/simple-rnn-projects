# Model Card: Neural Skip-gram Word Embeddings

## Model summary

The primary model is a compact PyTorch skip-gram network. It uses an `Embedding` layer
to convert a center-word index into a **32-dimensional**
dense vector and a linear output layer to predict nearby context words.

```text
Center word index
        ↓
PyTorch Embedding layer (32 dimensions)
        ↓
Linear vocabulary projection
        ↓
Softmax context-word probabilities
```

## Training data

- 1,020 synthetic, privacy-safe sentences
- 9,306 processed tokens
- Five curated domains: AI, finance, healthcare, energy, quality/manufacturing
- No personal, customer, or employer data

## Training configuration

- Vocabulary size: 171
- Context window: 2 words on each side
- Training center/context pairs: 26,438
- Validation pairs: 4,666
- Loss: cross-entropy
- Optimizer: Adam
- Early stopping: validation loss
- Random seed: 42

## Evaluation

| Metric | Result |
|---|---:|
| Validation loss | 3.3070 |
| Validation perplexity | 27.30 |
| Validation top-1 context accuracy | 20.77% |
| Validation top-5 context accuracy | 53.75% |
| Neural embedding domain purity@5 | 94.34% |
| TF-IDF + SVD baseline domain purity@5 | 80.61% |

Domain purity@5 measures how often a curated domain word's five nearest neighbors belong
to the same domain. It is useful for this controlled educational corpus but is not a
general benchmark of language understanding.

## Intended use

- Educational explanation of dense word representations
- Portfolio demonstration of neural representation learning
- Nearest-neighbor and vector-space exploration
- Sentence-vector and semantic-search prototypes
- Foundation for understanding embedding inputs to Simple RNN models

## Out-of-scope use

Do not use this model for production search, automated decisions, fairness conclusions,
medical advice, financial advice, or semantic interpretation of sensitive text.

## Limitations

- Small synthetic corpus
- Limited vocabulary and out-of-vocabulary coverage
- Context-independent word vectors
- Mean-pooled sentence embeddings discard word order
- Curated topic structure makes domain-purity evaluation easier than open-domain language
- Learned relationships reflect the training corpus and should not be treated as definitions
