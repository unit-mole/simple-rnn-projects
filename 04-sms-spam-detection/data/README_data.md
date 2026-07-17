# SMS Spam Dataset Guidance

## Dataset provenance

The original notebook downloads a tab-separated mirror of the public SMS Spam
Collection from:

```text
https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
```

The mirror contains two columns:

```text
label
message
```

Labels are `ham` and `spam`.

The authoritative dataset record is maintained by the UCI Machine Learning
Repository:

- **Dataset:** SMS Spam Collection
- **Creators:** Tiago Almeida and José María Gómez Hidalgo
- **DOI:** `10.24432/C5CC84`
- **License:** Creative Commons Attribution 4.0 International

Recommended citation:

> Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset].
> UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

The project retains the mirror used by the supplied notebook so the published
training result remains reproducible.

## Portfolio cleaning result

| Dataset stage | Rows |
|---|---:|
| Raw messages | 5,572 |
| Blank normalized messages removed | 2 |
| Duplicate normalized messages removed | 469 |
| Modeling rows | 5,101 |
| Ham messages | 4,499 |
| Spam messages | 602 |

Normalized duplicates are removed **before** splitting. This prevents the same
cleaned message from appearing in training, validation, and test partitions.

## Modeling split

```text
Training:   3,570 messages
Validation:   765 messages
Testing:      766 messages
```

All partitions are stratified. The test set is untouched until final evaluation
and threshold selection uses validation data only.

## Included sample

```text
data/sample_sms_messages.csv
```

The sample messages are hand-written and privacy-safe. The
`illustrative_category` column is descriptive only and is not used by the model.

## Supported upload columns

Message columns:

```text
message
text
sms
body
content
v2
```

Optional label columns:

```text
label
target
class
category
v1
```

## Privacy

Do not commit or upload private conversations, personal identifiers,
authentication codes, financial or health information, employer or customer
messages, or secrets.
