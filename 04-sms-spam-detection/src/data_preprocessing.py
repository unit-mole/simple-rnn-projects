"""Dataset loading, schema detection, and label normalization."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd

from .config import (
    LABEL_COLUMN_CANDIDATES,
    MAX_MESSAGE_CHARACTERS,
    MESSAGE_COLUMN_CANDIDATES,
)
from .text_preprocessing import clean_text

def identify_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

def identify_message_column(frame: pd.DataFrame) -> str:
    column = identify_column(frame.columns, MESSAGE_COLUMN_CANDIDATES)
    if column is None:
        raise ValueError(
            "No SMS message column was found. Use one of: "
            + ", ".join(MESSAGE_COLUMN_CANDIDATES)
        )
    return column

def identify_label_column(frame: pd.DataFrame) -> str | None:
    return identify_column(frame.columns, LABEL_COLUMN_CANDIDATES)

def normalize_binary_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("SMS label cannot be missing.")
    if isinstance(value, (int, float)) and float(value) in {0.0, 1.0}:
        return int(value)
    text = str(value).strip().lower()
    if text in {"0", "ham", "legitimate", "not spam", "non-spam"}:
        return 0
    if text in {"1", "spam", "junk"}:
        return 1
    raise ValueError(f"Unsupported binary label: {value!r}. Use ham/spam or 0/1.")

def clean_sms_frame(
    frame: pd.DataFrame,
    message_column: str | None = None,
    label_column: str | None = None,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict]:
    if frame is None or frame.empty:
        raise ValueError("The SMS dataset is empty.")

    message_column = message_column or identify_message_column(frame)
    label_column = label_column or identify_label_column(frame)
    selected = [message_column] + ([label_column] if label_column else [])
    cleaned = frame[selected].copy()

    rename_map = {message_column: "message"}
    if label_column:
        rename_map[label_column] = "label"
    cleaned = cleaned.rename(columns=rename_map)

    input_rows = len(cleaned)
    cleaned["message"] = cleaned["message"].fillna("").astype(str).str.strip()
    cleaned["clean_message"] = cleaned["message"].map(clean_text)

    blank_rows = int(cleaned["clean_message"].eq("").sum())
    cleaned = cleaned[cleaned["clean_message"].ne("")].copy()

    over_limit_rows = int((cleaned["message"].str.len() > MAX_MESSAGE_CHARACTERS).sum())
    if over_limit_rows:
        raise ValueError(
            f"{over_limit_rows} message(s) exceed the "
            f"{MAX_MESSAGE_CHARACTERS:,}-character limit."
        )

    duplicate_rows = int(cleaned.duplicated(subset=["clean_message"]).sum())
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates(subset=["clean_message"], keep="first")

    if "label" in cleaned.columns:
        cleaned["label"] = cleaned["label"].map(normalize_binary_label)
        cleaned["class_name"] = cleaned["label"].map({0: "ham", 1: "spam"})

    cleaned = cleaned.reset_index(drop=True)
    report = {
        "input_rows": int(input_rows),
        "output_rows": int(len(cleaned)),
        "blank_messages_removed": blank_rows,
        "duplicate_messages_found": duplicate_rows,
        "duplicates_removed": bool(drop_duplicates),
        "message_column": message_column,
        "label_column": label_column,
    }
    return cleaned, report

def load_sms_csv(path_or_buffer: str | Path | object) -> tuple[pd.DataFrame, dict]:
    return clean_sms_frame(pd.read_csv(path_or_buffer))

def load_tab_separated_sms(path_or_url: str | Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(
        path_or_url, sep="\t", header=None, names=["label", "message"]
    )
    return clean_sms_frame(frame, message_column="message", label_column="label")
