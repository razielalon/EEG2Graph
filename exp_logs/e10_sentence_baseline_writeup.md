# E10 — Sentence-level baselines & cross-dataset replication (RQ1)

**Status: done (post-hoc, CPU, both ZuCo datasets).** Two REBEL-free experiments
that turn the negative result into a *quantified* one: (a) what triplet-F1 does the
sentence-identity signal actually buy, used directly? and (b) does the core RQ1
finding replicate on independent data?

## E10a — Sentence-level retrieve-and-copy baseline

**Question:** E4/E9 show the only decodable EEG signal is *sentence identity*, not
word-level triplet structure. So use that signal in the most direct possible way —
retrieve the nearest sentence from the EEG, **emit that sentence's gold triplets**,
score triplet-F1 — and ask: does that beat the trained generative model, and does it
*work*? Method: E4/E9's exact retrieval front end (`retrieval_topk`'s StandardScaler
→ Ridge → TF-IDF/SVD map), then copy the top-1 sentence's triplets and score with the
same micro-averaged exact-match F1 the model is graded on. Script:
`model/sentence_retrieval_baseline.py`. Data: `exp_logs/e10_sentence_baseline_{zuco2,zuco1}.json`.

| condition (test triplet-F1) | ZuCo 2.0 | ZuCo 1.0 |
|---|---|---|
| generative model (E5 best cell) | 0.0083 | — |
| most-frequent-train triplet *(the model's collapse mode, E3)* | 0.0000 | 0.0000 |
| random-sentence copy *(chance for this metric, 20 draws)* | 0.0142 | 0.0165 |
| **EEG oracle retrieve-and-copy** *(gallery = test sentences)* | **0.0674** | **0.0941** |
| EEG deployable retrieve-and-copy *(gallery = train sentences)* | 0.0000 | 0.0000 |
| self-copy sanity *(copy the true sentence's triplets)* | 1.0000 | 1.0000 |

(EEG→sentence retrieval top-1 = 0.065 / 0.085; 143/1936 and 116/1224 gold triplets
recovered by the oracle.)

### Two conclusions, pulling opposite ways

**1. The generative architecture wastes the signal (~8–10×).** The oracle
retrieve-and-copy baseline scores **~8× the trained model** on ZuCo 2.0 (0.067 vs
0.008) and ~6× the random-copy floor, and the same pattern holds on ZuCo 1.0 (0.094).
The sentence-identity signal the EEG carries — and which E9 showed *survives into the
model's own representation* — is worth an order of magnitude more triplet-F1 than the
word-level generative pipeline extracts from it. The model does not fail for lack of
signal; it fails because a generative, word-level decoder cannot exploit a
sentence-level fingerprint the way a trivial retrieve-then-look-up can.

**2. But even the retrieval baseline does not *work* — the negative result is robust.**
Three facts keep this from being a fix: (i) 0.067–0.094 is still far below usable;
(ii) the oracle only scores above zero because it is *handed the candidate set* (the
test sentences) — the **deployable** version, which must retrieve among *training*
sentences because real test sentences are unseen (splits are by sentence), scores a
flat **0.0000**; and (iii) the model's own collapse mode (emit the most-frequent train
triplet) is also **0.0000**. On a sentence-disjoint split, triplets essentially never
recur across sentences, so nothing that is not already told the answer can produce
them. The signal is real but does **not** transfer to triplet generation on unseen
sentences — because the thing that would transfer is word-level structure, which E4
shows is absent.

The two together are the precise, defensible statement of the failure: *a weak
sentence-identity signal exists and reaches the decoder (E4/E9); the generative
architecture extracts ~1/10th of even that (E10a-1); and no method — generative or
retrieval — generalizes to unseen sentences, because word-level triplet structure is
what the task needs and what the EEG lacks (E4, E10a-2).*

## E10b — Cross-dataset replication of E4 (external validity)

**Question:** is "sentence-identity yes, word-structure no" a ZuCo-2 artifact? Re-run
E4's REBEL-free probe (`tests/eeg_signal_probe_extended.py`) on **ZuCo 1.0** — 12
different subjects (Z*), a different task pairing (NR + TSR), independently collected.
Data: `exp_logs/e4_signal_probes_zuco1.json`.

| metric | ZuCo 2.0 | ZuCo 1.0 |
|---|---|---|
| sentence retrieval, all bands (top-10) | 0.394 (2.6× chance) | **0.420 (2.7× chance)** |
| chance (10 / #unique test sents) | 0.149 | 0.156 |
| strongest single band | gamma (g2 0.332) | gamma (g1 0.424) |
| frequency trend | rises θ→γ | rises θ→γ |
| content-vs-function acc (majority) | 0.696 (0.699) → chance | 0.737 (0.738) → chance |
| word-length R² | −0.018 → none | −0.017 → none |

**Conclusion.** The RQ1 result replicates almost exactly on independent data:
sentence identity is decodable at ~2.6–2.7× chance, broadband and gamma-strongest,
while word-level structure (content/function, word length) sits at chance on both
datasets. If anything ZuCo 1.0's sentence signal is marginally stronger. This upgrades
RQ1 from a single-dataset observation to a cross-dataset finding, and closes the
"maybe it's a ZuCo-2 quirk" objection.

## What E10 does **not** show (honest limits)

- The **oracle** number is a transductive *ceiling*, not a deployable system — it is
  handed the candidate set. The deployable and generalization-relevant number is
  0.0000. Report 0.067/0.094 as "what sentence identity buys in the best case," never
  as a working result.
- E10a uses **raw EEG** (the signal in the *data*), complementary to E9 (the signal in
  the *trained model*). A bridge-representation version of the retrieve-and-copy
  baseline could be added on the cluster, but E9 already establishes the bridge retains
  the signal, and the deployable=0 result would be unchanged.
- The word-class nulls (E10b) use the same fixated-word probe as E4; ZuCo-1 channel
  study was skipped (montage-free, not needed for the granularity claim).

## Where this points

E10 is the constructive capstone of the RQ1 arc: it quantifies what the sentence-level
signal is worth in the task's own metric (little, ~0.07–0.09, and only transductively),
shows the generative model throws away ~90% of even that, and confirms the whole RQ1
picture on a second dataset. Combined with E9 (signal preserved in the model but
representationally mismatched), the thesis can now say precisely *why* F1≈0 and *what
the signal can and cannot support* — rather than only that it fails.

## Reproduction

`model/sentence_retrieval_baseline.py --data processed_zuco --out exp_logs/e10_sentence_baseline_zuco2.json`
(and `--data processed_zuco1`), REBEL-free, needs `<data>/sentence_triplets.json`.
E10b: `python tests/eeg_signal_probe_extended.py <processed_zuco1> --skip_channels`
(E4's script, from `exp/tier1-conclusions`).
