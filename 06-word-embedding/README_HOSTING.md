# Hosting Guide

## Recommended platform

Use **Streamlit Community Cloud**. The application is already written in Streamlit,
requires no secrets, loads compact saved artifacts, and does not retrain or download a
dataset during startup.

Official deployment documentation:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Required repository files

```text
06-word-embedding/
├── app/
│   ├── streamlit_app.py
│   └── requirements.txt
├── data/
│   ├── sample_text.csv
│   └── topic_lexicon.json
├── models/
│   ├── word_embedding_model.pt
│   ├── embedding_matrix.npy
│   ├── vocabulary.json
│   ├── model_metadata.json
│   └── training_config.json
├── outputs/
├── src/
└── requirements.txt
```

## Deployment configuration

Use these values when creating the app:

```text
Repository:      unit-mole/simple-rnn-projects
Branch:          main
Main file path:  06-word-embedding/app/streamlit_app.py
Python version:  3.12
```

No secrets are required.

## Deployment steps

1. Confirm that `06-word-embedding/` is committed to the `main` branch.
2. Sign in to Streamlit Community Cloud with the GitHub account that can access the repository.
3. Select **Create app** and choose deployment from GitHub.
4. Enter the repository, branch, and main file path shown above.
5. Open advanced settings and select Python 3.12.
6. Choose an available application URL.
7. Click **Deploy** and monitor the build logs.
8. Test all five application sections after the build completes.
9. Add the live URL to the project README and the repository-level README.

Streamlit creates a fresh environment and installs packages declared in
`app/requirements.txt`. Official dependency guidance:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Post-deployment checks

- Word Explorer returns nearest words.
- Sentence Embedding shows token indices and a 32-dimensional matrix.
- Semantic Search returns topic-relevant sentences.
- Model Performance loads all plots.
- GitHub and live-demo links work.
- The app logs contain no missing-file or import errors.

## Common issues

### `ModuleNotFoundError: No module named 'src'`

The app adds the project root to `sys.path`. Confirm that the deployed entrypoint is
exactly:

```text
06-word-embedding/app/streamlit_app.py
```

### Model file not found

Confirm that these files are visible on GitHub:

```text
06-word-embedding/models/word_embedding_model.pt
06-word-embedding/models/embedding_matrix.npy
06-word-embedding/models/vocabulary.json
```

### Dependency installation failure

Confirm that `06-word-embedding/app/requirements.txt` exists and contains the runtime
dependencies. Reboot the app after pushing corrections.

## Updating the deployed app

Changes pushed to the linked branch normally trigger an update. For dependency or
artifact changes, review the Streamlit logs and reboot the app when needed.
