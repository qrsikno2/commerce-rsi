# Amazon-C4 data audit

This report describes a real run on the complete downloaded Amazon-C4 files on
2026-10-07. Unit-test fixtures are separate from these results. The dataset is used
under a custom training/task protocol; this project does not reproduce the
dataset's original evaluation-only protocol.

Source revision: `39322697749a88d179f88d322a2fe4765b655c98`.
Acquisition URLs, transfer route, verification basis, sizes, and complete SHA256
values are recorded in `artifacts/data/raw/source_manifest.json`.
The query CSV was transferred from the official origin; the catalog bytes used a
mirror and were checked against metadata obtained from the official origin through
the server's existing proxy. The provenance manifest distinguishes those routes.

| Observation | Actual count |
| --- | ---: |
| Query records, all structurally complete | 21,223 |
| Unique qids | 21,223 |
| Unique normalized query texts | 21,222 |
| Unique target parent-ASINs | 20,463 |
| Unique users | 20,838 |
| Target IDs appearing in multiple queries | 614 |
| Users appearing in multiple queries | 348 |
| Catalog rows and unique item IDs | 1,058,417 |
| Catalog category values | 31 |
| Malformed query/catalog records | 0 |
| Missing target IDs | 0 |
| Catalog items with blank public text after stripping whitespace | 3,281 |
| Target items with blank public text | 90 |
| Queries whose target text is blank | 102 |

Target-ID coverage is 100%; coverage by targets with usable text is 99.5194% of
queries. The preparation stage must exclude the 102 queries with textless targets
and record the exclusion before constructing tasks. Category labels are used as
provided by this dataset; labels such as `Care` or `Cell` are not silently expanded
or merged into an invented taxonomy.

Grouping by shared user, target, and normalized query yields 20,079 connected
components: 19,132 singletons and a largest component of 13 queries (0.0613% of all
queries). There is no giant-component obstacle at this stage. Component counts
after textless-target removal can differ; cross-category components must be
quarantined intact when constructing category tasks.

The actual prepared protocol keeps 19,970 queries across 19 category tasks. It
excludes 102 queries with textless targets, 591 queries belonging to components
that cross category boundaries, and 560 queries from categories with fewer than
150 usable queries. These exclusions sum to the 1,253-query reduction from the raw
dataset. After target-text filtering there are 19,989 components; the largest
still contains 13 queries. Whole categories determine the outer split, and intact
components determine each task's inner split.

| Outer split | Category tasks | Business train | Reward validation | Final evaluation |
| --- | ---: | ---: | ---: | ---: |
| meta-train | 11 | 8,715 | 1,868 | 1,866 |
| meta-dev | 4 | 2,153 | 461 | 461 |
| meta-test | 4 | 3,112 | 668 | 666 |

An independent check of the produced files found zero shared-user,
shared-target-ID, or exact-normalized-query violations across the full
`(outer split, task, inner split)` assignment. All 57 query-file hashes and the
fixed 30,000-item public catalog hash were verified. All exported query records
use exactly `qid`, `query`, and `item_id`; public catalog records use exactly
`item_id`, `category`, and `text`. Every exported target belongs to the fixed
catalog. All meta-test query files and all final-evaluation files are under
`private/`. This is logical data separation; actual execution isolation still
requires a sandbox before any Agent-generated program can run.

The catalog SHA256 is
`16f102fca82174ecf0972f9a9aa8bfe47b137b5cc0447ceb76ca70ca45315f44`.
The same public item texts are intentionally shared between tasks, without target
membership or split labels. Catalog inclusion alone is not a claim of independent
products across tasks. The preparation manifest records exact query membership,
exclusions, hashes, and detection limits.

There is one exact duplicate query pair. The near-duplicate audit verified 5,632
candidate pairs from deterministic 64-permutation MinHash LSH over normalized word
trigrams. It found no additional pairs at Jaccard >= 0.85, and no bucket or candidate
budget cap was reached. The preparation threshold is 0.90. Candidate generation
is approximate, so this result does not establish absence of paraphrases or all
near duplicates. Detected pairs must be joined before splitting.

Query lengths have median 191 characters / 37 whitespace tokens and 99th
percentile 755 characters / 138 tokens. Public product texts have median 199
characters / 32 tokens and 99th percentile 3,661 characters / 588 tokens. The
longest product text has 187,930 characters / 32,800 whitespace tokens, so training
must explicitly record its fixed truncation rule. Seven items exceed 100,000
characters; their IDs appear in the machine-readable anomaly report.

The audit checks forbidden columns only for header presence. Safe query records
contain only `qid`, `query`, `item_id`, and `user_id`; dictionary product metadata
uses only `title` and `description`. Reports contain counts, distributions, and
ID/line-number anomaly examples, with no query/review/rating text. The full catalog
is streamed through a temporary on-disk SQLite index with a 16 MiB page cache;
only the small query set and bounded near-duplicate candidate structures reside
in Python memory.

Detailed output: `artifacts/data/audit/audit.json` and
`artifacts/data/audit/AUDIT.md`. A reproducible invocation from the project root is:

```bash
PYTHONPATH=src .venv/bin/python -c 'from pathlib import Path; from commerce_rsi.data.audit import audit_dataset; audit_dataset(Path("artifacts/data/raw"), Path("artifacts/data/audit"))'
```

These are synthetic queries derived from positive reviews, with typically one
labeled parent-ASIN target. Other products remain unlabeled; the data cannot
establish full relevance, user satisfaction, or variant-level color/size accuracy.
No business-model performance or Agent-learning result is implied by this audit.
