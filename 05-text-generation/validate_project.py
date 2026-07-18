"""Validate the Text Generation project structure and saved inference artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_generator import generate_text, load_generation_bundle


REQUIRED_PATHS = [
    "README.md",
    "README_HOSTING.md",
    "PROJECT_REVIEW.md",
    "requirements.txt",
    "requirements-ci.txt",
    ".python-version",
    "app/streamlit_app.py",
    "app/requirements.txt",
    "data/README_data.md",
    "data/sample_prompts.csv",
    "data/sample_text.txt",
    "models/text_generation_simple_rnn_model.pt",
    "models/vocabulary.json",
    "models/model_metadata.json",
    "models/training_config.json",
    "models/MODEL_CARD.md",
    "notebooks/text_generation.ipynb",
    "notebooks/archive/text_generation_original.ipynb",
    "images/01_text_generation_interface.png",
    "images/02_generated_text_result.png",
    "images/03_model_performance_dashboard.png",
    "images/04_project_overview.png",
    "outputs/model_metrics.json",
    "outputs/training_history.csv",
    "outputs/training_curve.png",
    "outputs/temperature_comparison.png",
    "src/data_preprocessing.py",
    "src/text_preprocessing.py",
    "src/sequence_generation.py",
    "src/model_training.py",
    "src/model_evaluation.py",
    "src/text_generator.py",
    "src/visualization.py",
    "tests/test_preprocessing.py",
    "tests/test_generation.py",
    "tests/test_model_loading.py",
    "train_model.py",
]


def _check_json(relative_path: str) -> dict:
    path = PROJECT_ROOT / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def validate_project(*, write_report: bool = True) -> dict:
    checks: list[dict[str, object]] = []

    missing = [item for item in REQUIRED_PATHS if not (PROJECT_ROOT / item).exists()]
    checks.append(
        {
            "check": "required_project_files",
            "passed": not missing,
            "details": "All required files are present." if not missing else f"Missing: {missing}",
        }
    )

    try:
        metadata = _check_json("models/model_metadata.json")
        vocabulary = _check_json("models/vocabulary.json")
        metrics = _check_json("outputs/model_metrics.json")
        required_metadata = {
            "sequence_length",
            "vocabulary_size",
            "validation_loss",
            "validation_accuracy",
            "validation_perplexity",
        }
        metadata_ok = required_metadata.issubset(metadata)
        vocabulary_ok = int(vocabulary["vocabulary_size"]) == int(metadata["vocabulary_size"])
        metrics_ok = abs(float(metrics["validation_loss"]) - float(metadata["validation_loss"])) < 1e-9
        passed = metadata_ok and vocabulary_ok and metrics_ok
        details = (
            f"Vocabulary={metadata.get('vocabulary_size')}, sequence_length={metadata.get('sequence_length')}, "
            f"validation_loss={metadata.get('validation_loss'):.4f}"
            if passed
            else "Metadata, vocabulary, or metrics are inconsistent."
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        passed = False
        details = f"JSON validation failed: {exc}"
    checks.append({"check": "artifact_metadata_consistency", "passed": passed, "details": details})

    try:
        nbformat.read(PROJECT_ROOT / "notebooks/text_generation.ipynb", as_version=4)
        nbformat.read(PROJECT_ROOT / "notebooks/archive/text_generation_original.ipynb", as_version=4)
        passed = True
        details = "Both notebooks are valid nbformat v4 documents."
    except Exception as exc:  # pragma: no cover - diagnostic path
        passed = False
        details = f"Notebook validation failed: {exc}"
    checks.append({"check": "notebook_integrity", "passed": passed, "details": details})

    try:
        bundle = load_generation_bundle(PROJECT_ROOT / "models")
        generated = generate_text(
            bundle,
            seed_text="Alice was beginning",
            generation_length=40,
            temperature=0.7,
            top_k=20,
            random_seed=42,
        )
        passed = generated.startswith("Alice was beginning") and len(generated) >= 55
        details = f"Model loaded and generated {len(generated)} characters."
    except Exception as exc:  # pragma: no cover - diagnostic path
        passed = False
        details = f"Inference smoke test failed: {exc}"
    checks.append({"check": "saved_model_inference", "passed": passed, "details": details})

    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    required_sections = [
        "## Project Highlights",
        "## Application Preview",
        "## Project Status and Honest Scope",
        "## Model Results",
        "## Project Structure",
        "## Run Locally",
        "## Deployment",
        "## Known Limitations",
        "## Skills Demonstrated",
        "## Portfolio Description",
    ]
    absent_sections = [section for section in required_sections if section not in readme_text]
    checks.append(
        {
            "check": "readme_portfolio_sections",
            "passed": not absent_sections,
            "details": "README contains all portfolio sections."
            if not absent_sections
            else f"Missing sections: {absent_sections}",
        }
    )

    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "project": "05-text-generation",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
    }

    if write_report:
        (PROJECT_ROOT / "VALIDATION_REPORT.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write-report",
        action="store_true",
        help="Run validation without updating VALIDATION_REPORT.json.",
    )
    args = parser.parse_args()
    report = validate_project(write_report=not args.no_write_report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
