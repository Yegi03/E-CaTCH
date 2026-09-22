# Research directions (student-scoped)

Pick **one** primary track. Each is sized for a serious semester / independent study, not a thesis-length program—unless you expand it.

---

## Track A — Critical survey (lowest infra burden)

**Question:** What do recent (2020–2026) depression-detection papers claim, and how often do they meet validity/ethics bars?

**Method:**

1. Define inclusion: social-media or text-based depression detection; peer-reviewed or major arXiv+venue
2. Code each paper on a rubric adapted from Chancellor & De Choudhury (2020) + PROBAST-style bias items (see JBDS review):
   - Label type (clinical / questionnaire / self-disclose / community / crowd)
   - Platform & language coverage
   - Train/val/test discipline; imbalance handling
   - Transfer / external validation attempted?
   - Ethics reporting (IRB, consent, anonymization)
   - Claims vs evidence (screening tool? diagnosis? research-only?)
3. Synthesize gaps; optionally compare to the AAAI meta-analysis findings

**Deliverable:** survey paper or technical report + spreadsheet of coded studies.

**Stretch:** release the coding sheet as a living resource under this `misdiagnoses` project.

---

## Track B — Label audit / “computational misdiagnosis” case study (high impact)

**Question:** How often do popular weak labels or crowd symptom labels disagree with expert or questionnaire-grounded judgments?

**Method (example):**

1. Reproduce or re-analyze a public symptom dataset (e.g., PRIMATE-style PHQ-9 annotations; follow Data Use Agreements)
2. Design a small expert or advanced-student annotation protocol with clear symptom definitions and evidence spans
3. Measure false positives/negatives by symptom (literature already flags anhedonia as hard)
4. Retrain a simple baseline on original vs cleaned labels; report metric shifts

**Deliverable:** short empirical paper: “label noise as misdiagnosis.”

**Ethics:** IRB/ethics board as required; no new scraping of identifiable users without approval.

---

## Track C — PHQ-9–grounded generalization experiment (ML-heavy)

**Question:** Does symptom grounding reduce cross-dataset failure the way Nguyen et al. (2022) suggest, and under what ablations?

**Method:**

1. Reimplement or adapt PHQ-9–constrained vs unconstrained classifiers
2. Evaluate in-domain vs transfer across ≥2 datasets
3. Error analysis: which symptoms drive false positives? figurative language? negation?
4. Optional: compare LLM few-shot prompting vs fine-tuning (subtype/classification papers are emerging—treat clinical claims cautiously)

**Deliverable:** experiment report with negative results welcome (negative transfer findings are valuable here).

---

## Track D — Clinical–computational bridge essay (health-interest student)

**Question:** How should computational “depression” labels relate to known clinical misdiagnosis patterns (esp. MDD vs bipolar spectrum)?

**Method:**

1. Narrative review of bipolar–MDD differential and over/underdiagnosis literature
2. Map those failure modes onto ML failure modes (binary collapse, missing longitudinal course, no hypomania signal in posts)
3. Propose evaluation requirements for any system claiming clinical utility (longitudinal features, differential outputs, human-in-the-loop, non-diagnostic framing)

**Deliverable:** interdisciplinary review aimed at HCI / digital health / CLPsych audiences.

**Caution:** Do not invent clinical guidelines; cite psychiatry sources and frame recommendations as research priorities.

---

## Track E — Risk & deployment analysis (policy / STS flavor)

**Question:** If a high-F1 depression detector shipped in a campus app or moderation tool, what harms follow from error types?

**Method:** scenario analysis using the FAT* taxonomy; interview synthesis (optional); map GDPR/HIPAA-style constraints at a high level; argue for red lines (no covert inference; no employment/insurance use).

**Deliverable:** position paper + checklist for “responsible non-deployment.”

---

## Recommendation for *this* student profile

Given interest in **health / diagnosing problems** and advisor interest in **how depression detection can be problematic**:

1. Start with Track A or D for orientation (2–3 weeks of reading)
2. Commit to **Track B or C** if they want an empirical contribution
3. Use Track E only as a chapter/section unless the student is stronger in ethics/policy than ML

**Default package we suggest:** Track A survey (rubric) + a small Track B pilot on one symptom (anhedonia or sleep) → enough for a workshop paper (e.g., CLPsych) if executed carefully.

---

## Out of scope (for now)

- Building a production screening product
- Claiming model outputs are diagnoses
- Scraping private messages or identifiable patient records without formal ethics approval
- Suicidality intervention systems without clinical partners and safety protocols
