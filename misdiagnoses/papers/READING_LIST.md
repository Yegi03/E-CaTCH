# Reading list: depression detection & misdiagnosis

Curated for a student project. Prefer reading the **must-read** set first (~5 papers), then branch by track.

Links are to official pages or open PDFs when available.

---

## Must-read (computational + ethics)

### 1. Methods critique — construct validity

**Chancellor, S., & De Choudhury, M. (2020).** Methods in predictive techniques for mental health status on social media: a critical review. *npj Digital Medicine*, 3, 43.  
https://doi.org/10.1038/s41746-020-0233-7 · https://www.nature.com/articles/s41746-020-0233-7

**Why it matters:** Best single starting paper. Reviews dozens of studies and shows weak construct validity, incomplete reporting, and limited clinical grounding. Frames *why leaderboard accuracy is not the same as detecting depression*.

**Student takeaway:** Use their criteria (validity, reproducibility, clinical applicability) as a checklist when reviewing newer papers.

---

### 2. Ethics taxonomy

**Chancellor, S., Birnbaum, M. L., Caine, E. D., Silenzio, V. M. B., & De Choudhury, M. (2019).** A taxonomy of ethical tensions in inferring mental health states from social media. In *FAT\* '19*.  
https://doi.org/10.1145/3287560.3287587 · PDF: http://www.munmund.net/pubs/FAT*2019_EthicsTaxonomy.pdf

**Why it matters:** Organizes consent/IRB gaps, validity/ML issues, and stakeholder harms (patients, clinicians, platforms, bad actors). Essential if the project argues that detection can be *dangerous*, not just inaccurate.

---

### 3. Generalization failure under weak labels

**Harrigian, K., Aguirre, C., & Dredze, M. (2020).** Do models of mental health based on social media data generalize? In *Findings of EMNLP 2020*.  
https://doi.org/10.18653/v1/2020.findings-emnlp.337 · https://aclanthology.org/2020.findings-emnlp.337/

**Why it matters:** Empirically shows large drops when transferring depression classifiers across platforms, plus confounds that inflate reported performance. Direct evidence for “high accuracy ≠ reliable detection.”

**Companion:** Harrigian, Aguirre, & Dredze (2021). On the state of social media data for mental health research. https://arxiv.org/abs/2011.05233 (dataset directory / meta view of available data).

---

### 4. Clinically grounded modeling (constructive path)

**Nguyen, T., Yates, A., Zirikly, A., Desmet, B., & Cohan, A. (2022).** Improving the generalizability of depression detection by leveraging clinical questionnaires. In *ACL 2022*.  
https://doi.org/10.18653/v1/2022.acl-long.578 · https://aclanthology.org/2022.acl-long.578/

**Why it matters:** Shows that constraining models with **PHQ-9 symptoms** improves out-of-distribution transfer vs. standard BERT. A concrete “do better” paper after the critique papers.

---

### 5. Annotation quality / false labels

**Milintsevich, K., Sirts, K., & Dias, G. (2024).** Your model is not predicting depression well and that is why: a case study of PRIMATE dataset. https://arxiv.org/abs/2403.00438

**Why it matters:** Mental-health professional re-annotation finds many false positives in crowd-labeled PHQ-9 symptom data (especially anhedonia). Shows that even “symptom-annotated” datasets can encode misdiagnosis-like noise.

---

## Strong recent surveys / meta-analyses

### 6. Meta-analysis of social-media depression AI (2022–2025 corpus)

**Tang, X., Jin, J. W., Ma, E., & Li, X. (2026).** AI in the Wild: A Meta-Analytic Evaluation of Depression Detection from Social Media Data. *Proceedings of the AAAI Conference on Artificial Intelligence*.  
https://doi.org/10.1609/aaai.v40i46.41283 · https://ojs.aaai.org/index.php/AAAI/article/view/41283

**Why it matters:** Updates the field: pooled performance looks strong, but label provenance (~75% weak supervision), ethics reporting (few IRBs), and explainability remain weak. Proposes a more clinically aligned evaluation framing (e.g., PHQ error, temporal stability, clinician agreement).

---

### 7. Systematic review of biases in ML depression detection

**Cao, Y., Dai, J., Wang, Z., Zhang, Y., Shen, X., Liu, Y., & Tian, Y. (2025).** Machine Learning Approaches for Depression Detection on Social Media: A Systematic Review of Biases and Methodological Challenges. *Journal of Behavioral Data Science*, 5(1), 67–102.  
https://doi.org/10.35566/jbds/caoyc · https://jbds.isdsa.org/jbds/article/view/110

**Why it matters:** Uses PROBAST-style risk-of-bias thinking. Documents Twitter/English/US–Europe skew, non-probability sampling, weak handling of negation/imbalance, and inconsistent tuning/splits. Good template for a student *bias audit* of a paper corpus.

---

## Clinical misdiagnosis bridge (health, not only NLP)

These papers ground why “wrong depression label” matters outside ML.

### 8. Bipolar vs unipolar depression (underdiagnosis)

**Phillips, M. L., & Kupfer, D. J. (2013).** Bipolar disorder diagnosis: challenges and future directions. *The Lancet* (see PMC reprint).  
https://pmc.ncbi.nlm.nih.gov/articles/PMC5858935/

**Why it matters:** Classic clinical framing: bipolar illness is often first (and long) treated as unipolar depression; delay of years; wrong meds can worsen course. Bridges student interest in “health diagnosing problems” to high-stakes differential diagnosis.

### 9. Over- and under-diagnosis of bipolar disorder

**Ghouse et al. / related systematic discussions** — start with:  
**Zimmerman, M.** literature on overdiagnosis; overview: *Overdiagnosis of Bipolar Disorder: A Critical Analysis of the Literature* — https://pmc.ncbi.nlm.nih.gov/articles/PMC3856145/

Plus clinician-oriented: *Differential Diagnosis of MDD Versus Bipolar Disorder* — https://www.psychiatrist.com/jcp/the-differential-diagnosis-of-mdd-versus-bp/

**Why it matters:** Shows misdiagnosis is bidirectional (false bipolar *and* missed bipolar). Useful if the student argues that simplified binary “depressed / not” models collapse clinically distinct entities.

---

## Optional deepening (pick 1–2)

| Paper | Angle |
|-------|--------|
| Yadav et al. D2S / PHQ-9 tweet symptoms (arXiv:2011.06149) | Symptom-level detection + figurative language |
| Zirikly & Dredze (CLPsych) explainability via PHQ auxiliary tasks | Interpretability for clinicians |
| Benton, Coppersmith, Dredze — ethical protocols for social media health research | Practical ethics checklist |
| De Choudhury et al. early depression Twitter work (foundational positive result papers) | Historical baseline; read *critically* after Chancellor 2020 |

---

## How to use this list

1. Read **1 → 2 → 3** in order (problem existence proof).
2. Read **4** and **5** (what “better” and “label noise” look like).
3. Skim **6** or **7** for an up-to-date landscape.
4. Read **8** (and optionally **9**) if connecting to clinical misdiagnosis / student health interest.

Then pick a research direction in [../docs/RESEARCH_DIRECTIONS.md](../docs/RESEARCH_DIRECTIONS.md).
