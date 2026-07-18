# Hosting Guide — Character-Level Text Generation using Simple RNN

## Recommended Platform

Use **Streamlit Community Cloud** for the first live deployment. The application already uses Streamlit, the committed PyTorch model is compact, and visitors can generate text without retraining the model.

## Repository Placement

Your existing monorepo should contain:

```text
simple-rnn-projects/
├── .github/
│   └── workflows/
│       └── text-generation-rnn-ci.yml
└── 05-text-generation/
    ├── app/
    │   ├── streamlit_app.py
    │   └── requirements.txt
    ├── data/
    ├── models/
    ├── outputs/
    ├── src/
    ├── README.md
    └── requirements.txt
```

The Streamlit entrypoint is:

```text
05-text-generation/app/streamlit_app.py
```

`app/requirements.txt` is included so Streamlit Community Cloud can discover dependencies beside the application entrypoint. The project-level `requirements.txt` remains the canonical local dependency file.

## Files Required for Inference

Commit these files:

```text
05-text-generation/app/streamlit_app.py
05-text-generation/app/requirements.txt
05-text-generation/src/
05-text-generation/models/text_generation_simple_rnn_model.pt
05-text-generation/models/vocabulary.json
05-text-generation/models/model_metadata.json
05-text-generation/models/training_config.json
05-text-generation/data/sample_prompts.csv
05-text-generation/outputs/training_curve.png
05-text-generation/outputs/temperature_comparison.png
05-text-generation/outputs/baseline_comparison.csv
05-text-generation/.streamlit/config.toml
```

No API keys or external services are required.

## 1. Test Locally

From the repository root:

```bat
cd 05-text-generation
if not exist "C:\venvs" mkdir "C:\venvs"
python -m venv "C:\venvs\textgen"
call "C:\venvs\textgen\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -r requirements-ci.txt
python -m pytest -q
python validate_project.py
python -m streamlit run app\streamlit_app.py
```

Confirm that:

- The saved model loads without retraining.
- Sample and custom prompts work.
- Temperature and top-k change the output.
- The generated text can be downloaded.
- The performance and overview tabs render.
- The responsible-use warning is visible.

## 2. Commit the Updated Project

From the root of `simple-rnn-projects`:

```bash
git status
git add .github/workflows/text-generation-rnn-ci.yml 05-text-generation
git commit -m "Add portfolio-ready Simple RNN text generation project"
git push origin main
```

After the push, open the Actions tab and confirm that **Text Generation Simple RNN CI** passes.

## 3. Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud using GitHub.
2. Select **Create app**.
3. Choose the `unit-mole/simple-rnn-projects` repository.
4. Select the `main` branch.
5. Enter the main file path:

```text
05-text-generation/app/streamlit_app.py
```

6. Keep the default application URL or choose an available custom subdomain.
7. Do not add secrets because this project does not require them.
8. Select **Deploy**.
9. Review the build logs and confirm that PyTorch, Streamlit, NumPy, pandas, and Matplotlib install successfully.

## 4. Verify the Live Application

Test:

- At least two sample prompts
- A custom prompt
- Temperatures `0.3`, `0.7`, and `1.1`
- Multiple top-k settings
- Generated-text download
- Model-performance charts
- Mobile and desktop layout

The values displayed in the app should match `models/model_metadata.json`, and the model checksum validation should pass during startup.

## 5. Live Application URL

The deployed application is available at:

```text
https://simple-rnn-projects-72u2s8vhngrexwwgbjpy6r.streamlit.app/
```

Share this link through:

- GitHub repository description
- GitHub pinned repository
- Resume project section
- LinkedIn Featured section
- Personal portfolio website

Suggested label:

```text
Live Demo — Character-Level Text Generation using Simple RNN
```

## Deployment Maintenance

Changes pushed to the configured branch and project files automatically trigger a Streamlit rebuild. Before pushing model updates:

```bash
python train_model.py
python -m pytest -q
python validate_project.py
git add models outputs
git commit -m "Refresh text generation model artifacts"
git push origin main
```

## Troubleshooting

### Model artifacts are missing

Run:

```bash
python train_model.py
python validate_project.py
```

Commit the updated `models/` files before redeploying.

### Import error for `src`

Do not move `app/streamlit_app.py` outside the provided structure. The application resolves the project root relative to its own path.

### Dependency file is not found

Keep `app/requirements.txt` beside the Streamlit entrypoint. It mirrors the pinned project-level runtime dependencies used by the deployed application.

### Deployment takes too long

The model is intended for CPU inference and is loaded once through `st.cache_resource`. Do not add training to application startup.

### GitHub rejects the model file

Check the checkpoint size before committing. The included model is intentionally compact. Larger experimental checkpoints should be kept outside Git or managed through an appropriate model registry.

### Application output looks repetitive

Increase temperature moderately or increase top-k. Very low temperature deliberately concentrates probability on the most likely characters.

## Alternative: Hugging Face Spaces

A Docker-based Hugging Face Space can also host the Streamlit application. This provides model-centric visibility but requires a Dockerfile and more configuration. Streamlit Community Cloud remains the simpler first deployment option for this portfolio project.
