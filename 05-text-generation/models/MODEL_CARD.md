# Model Card — Character-Level Simple RNN

## Model summary

This project contains a compact character-level text-generation model implemented with PyTorch `nn.RNN`. It predicts the next character from the preceding 50-character context and generates text autoregressively.

## Intended use

- Educational sequence-modeling demonstration
- GitHub and portfolio presentation
- Interactive Streamlit inference demo
- Comparison with simple Markov or n-gram baselines

It is **not** intended for production content generation, factual writing, decision-making, or generation of private, harmful, or misleading content.

## Architecture

```text
Character IDs
    → Embedding (32)
    → Simple RNN (64 tanh units)
    → Dropout (0.20)
    → Dense vocabulary logits (71 characters)
```

## Training data

The included corpus is a compact public-domain excerpt from *Alice's Adventures in Wonderland* by Lewis Carroll. See [`../data/README_data.md`](../data/README_data.md).

## Evaluation

| Metric | Result |
|---|---:|
| Validation loss | 1.9543 |
| Validation next-character accuracy | 44.09% |
| Validation perplexity | 7.06 |

The corpus is split chronologically before overlapping sequence windows are created to reduce train-validation leakage.

## Limitations

- Simple RNNs have weak long-term memory.
- Character-level generation may produce incomplete words and grammar errors.
- The compact single-domain corpus limits generalization.
- Sampling behavior is sensitive to temperature and top-k settings.
- The model learns corpus patterns, not factual knowledge or human-level understanding.

## Responsible use

Generated text must be reviewed by a human. Do not use the model to create harmful, misleading, private, sensitive, or high-stakes content.
