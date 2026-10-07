# Recorded experiments

Generated from saved artifacts at 2026-10-07T15:47:33.182763+00:00.

Regenerate with `.venv/bin/python scripts/report_experiments.py`. The detailed JSON ledger is
`artifacts/experiment_summary.json`; it includes every source-file hash, failed attempt, update record, and measurement coverage.

These are development/control-path results. Final evaluation and meta-test remain locked. Unknown values are not zeros.

## Business-model runs

Metrics use completed runs only; costs include every saved run. Values are mean ± sample standard deviation across the listed paired-seed configuration; n=1 has unknown standard deviation.

| Series / task | Strategy | Steps | Towers | Seeds | MRR@10 | Full MRR | Total business seconds |
| --- | --- | ---: | --- | --- | --- | --- | ---: |
| paired_1000 / toys_7a17d0 | bm25 | 0 | independent (legacy field absent) | 17 | 0.039048 ± unknown (n=1) | unknown (n=0) | 4.00 |
| paired_1000 / toys_7a17d0 | uniform | 1000 | independent (legacy field absent) | 17 | 0.000000 ± unknown (n=1) | unknown (n=0) | 12.55 |
| standalone / toys_7a17d0 | uniform | 100 | independent (legacy field absent) | 17 | 0.001905 ± unknown (n=1) | unknown (n=0) | 6.38 |
| shared_100 / toys_7a17d0 | bm25 | 0 | shared | 17 | 0.039048 ± unknown (n=1) | unknown (n=0) | 4.02 |
| shared_100 / toys_7a17d0 | lexical_hard | 100 | shared | 17, 29, 43 | 0.001455 ± 0.001604 (n=3) | unknown (n=0) | 36.23 |
| shared_100 / toys_7a17d0 | uniform | 100 | shared | 17, 29, 43 | 0.001485 ± 0.001551 (n=3) | unknown (n=0) | 14.35 |
| paired_1000 / toys_7a17d0 | lexical_hard | 1000 | independent (legacy field absent) | 17, 29, 43 | 0.000000 ± 0.000000 (n=3) | unknown (n=0) | 52.90 |
| paired_1000 / toys_7a17d0 | uniform | 1000 | independent (legacy field absent) | 29, 43 | 0.002460 ± 0.000112 (n=2) | unknown (n=0) | 20.91 |

Configurations are separated by exact saved settings, data hashes, and code hash. Older run files without `shared_towers` came from the independent-tower implementation; they are not pooled with later shared-tower runs.

## Paired lexical-hard minus uniform differences

Only uniquely paired seeds with matching initialization and training-query schedules are included. Negative differences favor uniform for that metric.

| Series / task | Steps | Paired seeds | Δ MRR@10 | Δ full MRR | Excluded seed pairs |
| --- | ---: | --- | --- | --- | ---: |
| paired_1000 / toys_7a17d0 | 1000 | none | unknown (n=0) | unknown (n=0) | 1 |
| standalone / toys_7a17d0 | 100 | none | unknown (n=0) | unknown (n=0) | 1 |
| shared_100 / toys_7a17d0 | 100 | 17, 29, 43 | -0.000030 ± 0.002789 (n=3) | unknown (n=0) | 0 |
| paired_1000 / toys_7a17d0 | 1000 | 29, 43 | -0.002460 ± 0.000112 (n=2) | unknown (n=0) | 1 |

## Agent pilots

An `updated` count requires a saved update record with that status. Verified counts additionally require changed adapter hashes and exact reload agreement. No-update attempts still consume load/update time and remain in all costs.

| Pilot | Mode / task | Status | Groups | Valid / failed / pending completions | Updated / no-update | Hash + reload verified | Valid full MRR mean |
| --- | --- | --- | ---: | --- | --- | ---: | ---: |
| eval_A0 | evaluate / toys_7a17d0 | completed | 2 | 2 / 2 / 0 | 0 / 0 | 0 | 0.008236 |
| eval_A1 | evaluate / toys_7a17d0 | running | 2 | 0 / 2 / 0 | 0 / 0 | 0 | unknown |
| fixed_A0 | fixed / garden_03c966 | completed | 2 | 2 / 2 / 0 | 0 / 0 | 0 | 0.013229 |
| pilot_v1 | recursive / garden_03c966 | failed | 3 | 0 / 6 / 0 | 0 / 3 | 0 | unknown |
| pilot_v2 | recursive / garden_03c966 | completed | 2 | 2 / 2 / 0 | 2 / 0 | 2 | 0.013229 |

| Pilot | Generation load s | Generation s | Candidate execution s | Agent update total s | Update load s (included) | Optimizer s (included) | Completion tokens | Recorded wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eval_A0 | 7.46 | 91.81 | 58.92 | 0.00 | 0.00 | 0.00 | 1351 | 159.00 |
| eval_A1 | 3.69 [partial 1/2] | 56.42 [partial 1/2] | 6.74 [partial 2/3] | 0.00 | 0.00 | 0.00 | 832 [partial 1/2] | unknown |
| fixed_A0 | 20.89 | 110.07 | 51.31 | 0.00 | 0.00 | 0.00 | 1679 | 185.76 |
| pilot_v1 | 25.74 | 186.37 | 27.94 | 12.21 | 11.36 | 0.00 | 2913 | 256.26 |
| pilot_v2 | 21.72 | 99.33 | 50.72 | 17.78 | 7.28 | 1.55 | 1578 | 193.31 |
| All saved pilots, including failures | 79.49 [partial 10/11] | 543.99 [partial 10/11] | 195.63 [partial 20/21] | 29.99 | 18.64 | 1.55 | 8353 [partial 10/11] | 794.32 [partial 4/5] |

Business total/training, sampler, CUDA-event, and update-token submeasurements are retained in the JSON. Execution seconds include business work; update load and optimizer seconds are subsets of update total. Do not sum overlapping columns.

| Pilot | Generation peak allocated GiB | Business peak allocated GiB | Update peak allocated GiB |
| --- | ---: | ---: | ---: |
| eval_A0 | 6.054 | 0.125 | unknown |
| eval_A1 | 5.990 [partial 1/2] | 0.101 | unknown |
| fixed_A0 | 6.102 | 0.125 | unknown |
| pilot_v1 | 6.064 | 0.118 | 5.923 |
| pilot_v2 | 6.102 | 0.125 | 8.110 |

`eval_A0` recorded candidate failures:

- 2 × SandboxError: candidate_failed: TypeError: Population must be a sequence.  For dicts or sets, use sorted(d).

`eval_A1` recorded candidate failures:

- 1 × SandboxError: candidate_failed: TypeError: Population must be a sequence.  For dicts or sets, use sorted(d).
- 1 × SandboxError: candidate_failed: TypeError: sample_negatives() takes 4 positional arguments but 5 were given

`fixed_A0` recorded candidate failures:

- 1 × SandboxError: candidate_failed: TypeError: Random.choices() got an unexpected keyword argument 'replace'
- 1 × SandboxError: invalid_output: worker did not return one JSON object

`pilot_v1` termination: RuntimeError: No differential execution rewards within the bounded round; no new generation claimed

`pilot_v1` recorded candidate failures:

- 1 × SamplingError: Row 23: duplicate negatives
- 2 × SamplingError: Row 27: known positive sampled as negative
- 1 × SamplingError: Sampler must return one list per query
- 1 × SandboxError: candidate_failed: ModuleNotFoundError: No module named 'numpy'
- 1 × SandboxError: candidate_failed: TypeError: Population must be a sequence.  For dicts or sets, use sorted(d).

`pilot_v2` recorded candidate failures:

- 1 × SandboxError: candidate_failed: TypeError: Random.choices() got an unexpected keyword argument 'replace'
- 1 × SandboxError: invalid_output: worker did not return one JSON object

## Limits of the evidence

- Real business training and successful Agent parameter updates are distinct evidence levels.
- Two updated/reloaded adapters prove a control path, not recursive capability improvement or RISE/RSI superiority.
- This report does not perform final evaluation or unlock meta-test. It only reads saved development/pilot logs.
- No equal-total-compute comparison is inferred; controller arguments alone do not match actual groups, tokens, or update costs.
- Seed variability on one task is not independent-task generalization; paired standard deviations are descriptive.
- Earlier independent-tower and later shared-tower business runs have different configurations and must not be pooled.
- Full MRR and MRR@10 are distinct. Missing older full-MRR values remain unknown.
- Running or incomplete attempts may have generated tokens or consumed GPU time before a cost artifact exists; that unrecorded work is unknown, not free.
- Manual inspection found plausible unlabeled relevant alternatives and many Books/Kindle lexical negatives in the Toys baseline; see `REVIEW.md`. Single-target retrieval scores do not establish full relevance or user satisfaction.
