# Amazon-C4 sampler research protocol (pilot v1)

This is a custom research split of a synthetic-query dataset originally released
for evaluation. It is not a reproduction of the original Amazon-C4 benchmark.

Official dataset revision: `39322697749a88d179f88d322a2fe4765b655c98`.
Exact public candidate SHA256: `16f102fca82174ecf0972f9a9aa8bfe47b137b5cc0447ceb76ca70ca45315f44`.
Acquisition and exclusions are detailed in `DATA_AUDIT.md` and artifacts manifests.

## Split and access rules

- Components join shared users, targets, normalized queries and detected near
  queries (verified word-trigram Jaccard >= .9 after approximate MinHash candidates).
  Detection can miss near duplicates; it does not prove semantic independence.
- Components crossing product categories are quarantined whole. Categories with
  fewer than150 eligible queries are excluded.19 category tasks yield11 meta-train,
  4 meta-dev and4 meta-test tasks with seed20261007. No seed is treated as a new task.
- Within each task, intact groups are allocated approximately70/15/15 to business
  train/reward-validation/final-evaluation. All final splits and all meta-test query
  files are under `private/`. A trusted preparation audit sees them only to check
  isolation; they never enter sampler or Agent prompts, losses or feedback.
- Public corpus has30000 products including every usable target and deterministically
  selected distractors. All methods use the same corpus. Catalog text overlap is
  explicitly allowed; target mappings and split membership are not public features.
- Query export whitelist: `qid, query, item_id`. Public product whitelist:
  `item_id, category, text`. User IDs stay in private provenance. Source review/rating
  fields never enter exported inputs. Public metadata may itself contain quoted
  publisher/reviewer copy; this is distinct from the excluded source review fields.
- Exact normalized product-text aliases are excluded when sampling a known positive.
  Other unlabeled products may still be relevant, and edition/variant equivalence
  is incomplete. No variant-level accuracy or user-satisfaction claims are supported.

## Frozen pilot environment

See `configs/pilot.json`. A shared text tower uses hashed token embeddings, pooling,
MLP and normalized vectors, Adam and sampled softmax. Each candidate restarts the
same weights, query-batch schedule, seed17,100 steps,32 queries and16 negatives.
Current model weights always recompute product vectors before evaluation.

Initial meta-dev calibration (`artifacts/business_pilot/`) found very sparse
MRR@10. Before any Agent rollout, pilot primary reward was set to **full MRR**,
the mean reciprocal rank of the one labeled target over the fixed30000 products.
MRR@10, Hit@10 and NDCG@10 remain reported. This is a feedback-sparsity adjustment,
not a result on final evaluation. Execution failure gets0; there is no success
bonus, self-reported reward, silently repaired program, or fallback sampler.

Generated code sees only a JSON batch, public IDs/metadata (texts first256 characters),
step counters and a seeded stdlib RNG. Linux Landlock, seccomp, UID dropping and
resource limits protect the host trainer and answers. See `ISOLATION_FEASIBILITY.md`.

## Parameter learning and interpretation

The Qwen2.5-Coder-3B-Instruct base is frozen; rank8 q/v LoRA parameters train using
one on-policy RLOO policy-gradient step per group with different execution rewards.
Generation uses the exact untruncated policy distribution (temperature1, top_p1,
top_k0), dropout0. Prompt tokens are masked; completion log probabilities are
summed, including EOS. Leave-one-out reward baselines use independent completions
of one identical prompt. This backend is not upstream SLIME/Megatron and is not
called GRPO/GSPO. Equal rewards produce an explicit no-update result.

Every update verifies base+adapter identities against rollout provenance; saved
adapters are loaded again and compared by tensor hash. Later rounds load the prior
adapter and optimizer state from disk. Old A0 trajectories cannot update A1 here.

Pilot bounded budget: two intended generations, at most three groups of two
completions per generation,768 new tokens each,600s maximum business run each.
The OpenRSI program library and execution feedback persist across rounds, including
for the fixed-Agent control. All failed attempts, generation, business execution,
loads and optimizer work are counted separately in logs.

This stage proves or fails the control path. A single training task and a few
updates do not establish the RSI hypothesis. Formal conclusions require frozen
checkpoints, equal new-task search/token/compute budgets for A0/A1/A2, explicit
memory controls, paired task/seed uncertainty, and a single untouched meta-test
evaluation. The pilot controller cannot export or evaluate meta-test.
