# SMS Spam Dataset Guidance

## Source used by the supplied project

The original notebook downloads the public SMS Spam Collection from:

```text
https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
```

The tab-separated source contains `label` and `message` columns. Labels are
`ham` and `spam`.

## Portfolio cleaning result

| Dataset stage | Rows |
|---|---:|
| Raw messages | 5,572 |
| Blank normalized messages removed | 2 |
| Duplicate normalized messages removed | 469 |
| Modeling rows | 5,101 |
| Ham messages | 4,499 |
| Spam messages | 602 |

Duplicates are removed before splitting so normalized message text cannot appear
in more than one partition.

## Modeling split

```text
Training:   3,570 messages
Validation:   765 messages
Testing:      766 messages
```

All partitions are stratified. The test set is untouched until final evaluation.

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
