# Independent scientific and isolation review

Reviewed 2026-10-07: `agent_training/backend.py`, `data/prepare.py`,
`openrsi_adapter/sandbox.py`, `openrsi_adapter/upstream.py`, and the trusted retrieval
and sampler boundary. This is a bounded source review plus specific diagnostics,
not a formal proof of security or evidence of improved Agent capability.

## Findings and resolution

1. **Task role and catalog are not authenticated against the prepared manifest.**
   At review time, `read_task_package` validates the manifest hash but trusts
   `metadata.outer_split` and `metadata.catalog_sha256`. A read-only, in-memory
   Parquet-table diagnostic relabeled the real Toys meta-dev task as meta-train,
   retained its true data directory and prepared-manifest hash, and supplied an
   incorrect catalog hash. The reader accepted the row. The consumer must compare
   task membership, actual outer role, category/protocol, catalog hash, and expected
   directory against the authoritative prepared manifest. Agent weight updates
   must additionally require the authoritative meta-train role. No actual task
   package or dataset file was modified by the diagnostic. **Resolved:** the
   reader now checks the authoritative task's role/category, manifest
   protocol/catalog hash, deterministic UUID, and exact resolved task directory.
   Regression tests reject forged metadata and symlink escape.
2. **Nonfinite clipping configuration is not rejected.**
   `update_from_group` checks `max_grad_norm <= 0` but not finiteness. NaN therefore
   passes validation and can poison gradients through the clipping coefficient;
   an infinite value silently disables clipping. Require a finite positive value
   before model loading. **Resolved:** the backend now explicitly checks
   `math.isfinite(max_grad_norm)` and positive magnitude.
3. **Missing optimizer state silently starts a fresh optimizer.**
   If an input adapter is supplied but `optimizer.pt` is absent,
   `update_from_group` reports `optimizer_resumed=False` and proceeds. A continuation
   of the recorded recursive lineage should fail on missing state, or require an
   explicit, recorded optimizer-reset mode. Present state is hash- and
   lineage-validated correctly. **Resolved:** continuation now requires
   `optimizer.pt`; a real tiny CPU checkpoint test removes that file and confirms
   rejection instead of an implicit reset.

4. **Feedback labels initially disagreed with the pilot reward.** The pilot uses
   untruncated MRR, whereas the initial prompt/parent-feedback labels said MRR@10.
   **Resolved:** the prompt, task metadata, parent-feedback field, and task reader
   now consistently declare untruncated MRR. MRR@10/Hit@10/NDCG@10 remain reported
   secondary metrics. This pre-experiment metric adjustment must remain explicit
   in the protocol history.

The follow-up reviewed `recursion/pilot.py` and added nine CPU-only controller
contract tests. The selected controller/backend/upstream suite passed 29 tests
after these fixes. These tests use mocked business rewards for controller cases;
the separate tiny-backend test exercises real CPU parameter updates/reloads and
is explicitly not a real-task Agent experiment.

## Controller and comparison limits

The controller requests native upstream evolution prompts, loads the currently
selected adapter for each fresh rollout group, stores raw completions/tokens, and
extracts the source without rewriting it. Syntax failures and token-budget
truncations never reach execution; sandbox/business failures receive zero reward
and preserve available cost records. Successful rewards are trusted-host MRR.
Only the current group's rewards enter its RLOO update. Equal-reward groups do not
advance the adapter; exhaustion of the bounded attempts fails the round without
claiming a new generation. Fixed/evaluation modes never call the optimizer.

The recursive mode stops each round immediately after an update, while fixed and
evaluation modes consume every requested attempt. Consequently identical
`rounds` and `max_attempts` do **not** guarantee equal numbers of searches. Any
comparison must use the observed group/completion counts and separately report
Agent generation, business execution, and Agent-update costs. A proposed fresh
meta-dev group with two completions per A0/A1/A2 would be a bounded comparative
smoke test; it would not be a multi-task capability study or final meta-test.
No fixed-A0/once-trained-A1/recursive superiority conclusion follows from the
controller's existence or these unit tests.

Execution errors can mention host paths, but candidate code cannot open those
paths under the tested kernel sandbox. Parent programs and their feedback are
untrusted prompt content; downstream generated programs still enter the same
sampler-only execution boundary. Failure feedback observed in this review did
not expose private final-evaluation answer contents.

## Checks that support the current implementation

- RLOO uses the correct independent-sample leave-one-out baseline, completion-only
  causal log-probability sums including EOS, one update per sampled group, and no
  sequence-length normalization. Prompt/padding tokens are outside the loss.
  Policy, base, and adapter hashes are checked after reload; changed sampling
  distributions and mixed prompts are rejected. Equal rewards explicitly produce
  no update. This supports an on-policy RLOO description, not a GRPO/GSPO claim.
- Nonzero dropout is rejected; generation uses a fresh configuration with
  temperature 1 and unrestricted top-p/top-k. No old-policy trajectory reuse was
  observed in the backend. Actual reward-to-generated-token association remains
  the controller's responsibility.
- The worker executes candidate source only after dropping root privileges where
  applicable and installing Landlock plus an x86-64 seccomp allowlist. Files,
  network creation, subprocesses, tracing, and privilege changes are denied;
  input/output, memory, CPU, and wall time are bounded. Parent evaluation and
  output-ID/known-positive checking stay outside the worker. This is stronger than
  AST-only checks, but remains dependent on the Linux kernel's isolation behavior.
- During review, a concurrent fix replaced Python `-I` (which ignores
  `PYTHONHASHSEED`) with safe-path/no-site options and seeded the global random
  module. Three repeated executions of a set-order sampler with the same seed
  returned identical IDs. This verifies that example, not all programs that might
  deliberately consult clocks or OS entropy.
- The real prepared-data review found zero shared-user/target/normalized-query
  boundary violations, verified 57 query-file hashes and the 30,000-item catalog
  hash, and checked private placement and export whitelists. See `DATA_AUDIT.md`.
  JSONL must be iterated by file lines; `str.splitlines()` also splits Unicode
  separators that occur inside valid source strings. The retrieval loader already
  uses file-line iteration.

## Manual lexical-negative inspection

Inspected all five saved first-batch query examples (80 negative entries) from
`artifacts/business_pilot/paired_1000/lexical_hard_s17/negative_audit.jsonl`, their
training query text, and the corresponding public catalog metadata. This is a
small convenience sample, not a random estimate of false-negative prevalence.
No private final-evaluation labels were needed.
This saved baseline used the earlier independent-tower configuration; the later
Agent control-path pilot explicitly selects shared towers. The negative-inspection
notes describe that saved sample and should not be presented as a matched
performance comparison between the two configurations.

The 80 entries comprise 52 Kindle, 22 Books, 4 Toys, 1 Handmade, and 1 Cell item.
Thus 74/80 entries (92.5%) are Books/Kindle despite the Toys task. Many matching
texts contain publisher or reviewer blurbs with generic language resembling the
synthetic shopping queries. Those blurbs are part of the allowed public product
metadata, not exports of the banned `ori_review` column. Lexical rank therefore
does not imply a same-category or semantically difficult negative.

| Training qid | Selected item | Inspection and uncertainty |
| --- | --- | --- |
| 4977 | `B0C41HH3KK` (Farkle) | A plausible relevant alternative for the broad family/friends game request. The target is SEQUENCE, but the wording does not uniquely identify it. Treat as a possible false negative, not confirmed relevance. |
| 4977 | `B0BHF7S5KQ` (IYKYK) | Social party game matches part of the request, but public metadata specifies adults 18+ and adult content; the daughter's age is unknown. Relevance is uncertain. |
| 13595 | `B09MLWC5YX` (Scrabble Junior) | Matches simple word learning, but metadata says ages 5+ and does not establish Spanish instruction requested for a four-year-old. Partial semantic overlap with explicit mismatches. |
| 13595 | `B08BR7SF9Z` (tumble tower) | Family game for ages 3+ matches age/simple-play aspects, but not the requested word/Spanish-learning function. |
| 12182 | `0593135202` and `B08FHBV4ZX` (Project Hail Mary editions) | Both are unrelated to the stuffed-Pomeranian request and refer to the same novel in different formats. Their normalized full texts differ; exact-text de-duplication does not remove this semantic redundancy. |

No identical normalized full-text duplicate pair occurred among the inspected
negative lists. No sampled ID equaled its row's positive ID. The sample does not
establish a complete relevance judgment: only one parent-ASIN target is labeled,
and broad requests can admit other valid products. Do not reinterpret the manual
notes as new ground-truth labels or change a frozen evaluation pool based on them.
