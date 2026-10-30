# Resume target check

| Requirement | Status | Actual | Required |
|---|---|---|---|
| pytorch_repositories | met | 12 | 12 |
| recall_at_5 | met | 97.2% | 91% |
| status_accuracy | met | 87.3% | 87% |
| grounding_rate | met | 100.0% | 97% |

Retrieval on the regression set (top 5): claim2code_rerank hit 97.2%, standard recall 85.8%; hybrid_claim2code hit 72.7%; embedding_only hit 64.5%.
recall_at_5 uses the definition behind the archived 74%: an annotated symbol in the top 5.

Accuracy and retrieval use the unchanged original 79-claim regression set.
New positive coverage claims do not change that denominator.

Baseline comparison: met. The archived 74% (hybrid_claim2code, 79 claims, 8 pairs) is the evaluator's Recall@5 of that time: the share of claims with an annotated symbol in the top 5. The target is measured with the same definition on the same claims; standard per-symbol recall is reported alongside it.

All submitted resume claims supported: True.

Input-freezing advisories (development metrics remain in scope):

- retrieval: attention_is_all_you_need: paper SHA-256 is missing or a placeholder
- retrieval: bert: paper SHA-256 is missing or a placeholder
- retrieval: clip: paper SHA-256 is missing or a placeholder
- retrieval: ddim: paper SHA-256 is missing or a placeholder
- retrieval: ddpm: paper SHA-256 is missing or a placeholder
- retrieval: deit: paper SHA-256 is missing or a placeholder
- retrieval: detr: paper SHA-256 is missing or a placeholder
- retrieval: dino: paper SHA-256 is missing or a placeholder
- retrieval: dpo: paper SHA-256 is missing or a placeholder
- retrieval: lora: paper SHA-256 is missing or a placeholder
- retrieval: mae: paper SHA-256 is missing or a placeholder
- retrieval: moco: paper SHA-256 is missing or a placeholder
- retrieval: reflexion: paper SHA-256 is missing or a placeholder
- retrieval: sam: paper SHA-256 is missing or a placeholder
- retrieval: whisper: paper SHA-256 is missing or a placeholder
- verification: attention_is_all_you_need: paper SHA-256 is missing or a placeholder
- verification: bert: paper SHA-256 is missing or a placeholder
- verification: clip: paper SHA-256 is missing or a placeholder
- verification: ddim: paper SHA-256 is missing or a placeholder
- verification: ddpm: paper SHA-256 is missing or a placeholder
- verification: deit: paper SHA-256 is missing or a placeholder
- verification: detr: paper SHA-256 is missing or a placeholder
- verification: dino: paper SHA-256 is missing or a placeholder
- verification: dpo: paper SHA-256 is missing or a placeholder
- verification: lora: paper SHA-256 is missing or a placeholder
- verification: mae: paper SHA-256 is missing or a placeholder
- verification: moco: paper SHA-256 is missing or a placeholder
- verification: reflexion: paper SHA-256 is missing or a placeholder
- verification: sam: paper SHA-256 is missing or a placeholder
- verification: whisper: paper SHA-256 is missing or a placeholder
