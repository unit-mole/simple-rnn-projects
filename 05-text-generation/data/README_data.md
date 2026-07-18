# Dataset and Corpus Notes

## Included sample

`sample_text.txt` contains a compact excerpt from **Alice's Adventures in Wonderland** by Lewis Carroll, obtained from the Project Gutenberg plain-text edition used by the original project. The work is public domain in the United States and is included only as a small, reproducible demonstration corpus.

Source used by the original code:

```text
https://www.gutenberg.org/files/11/11-0.txt
```

The included sample covers Chapters I–IV and remains small enough for a portfolio repository. Review the copyright status in your own jurisdiction before redistributing or deploying a different corpus.

## Data format

The model expects a UTF-8 plain-text file. Punctuation, capitalization, and paragraph boundaries are retained because they provide useful character-level sequence information.

## Using another corpus

1. Place an allowed `.txt` file in this folder.
2. Run `python train_model.py --corpus data/your_file.txt`.
3. Commit only data you have permission to redistribute.
4. Keep private, copyrighted, or very large corpora outside Git by adding them to `.gitignore`.

## Sample prompts

`sample_prompts.csv` supplies safe seed prompts for the Streamlit interface. It does not participate in training.
