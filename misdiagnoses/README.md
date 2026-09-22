# Misdiagnoses

A student research project on **when diagnosis goes wrong**—especially automated and clinical depression detection—and what that implies for health AI and psychiatric practice.

## Motivation

Depression detection looks like a clean ML problem (binary labels, high F1 on Twitter/Reddit benchmarks). In practice it is fragile:

- Labels are often weak proxies (subreddit membership, keyword self-disclosure), not clinical diagnoses
- Models overfit platform and language artifacts and fail to transfer
- False positives risk stigma; false negatives miss people who need care
- Clinically, depression itself is frequently confused with bipolar spectrum illness and other conditions

This project collects the strongest papers and frames concrete research questions a student can pursue.

## Suggested focus for a student project

**Primary angle (recommended):** critique and improve *construct validity* in computational depression detection—not “build another classifier that gets 90% accuracy.”

Strong secondary angle: connect computational failures to **clinical differential diagnosis** (e.g., MDD vs bipolar II), where misdiagnosis already has a large evidence base.

See [docs/RESEARCH_DIRECTIONS.md](docs/RESEARCH_DIRECTIONS.md) for scoped project ideas (survey, re-annotation audit, PHQ-9 grounding experiment, ethics/risk analysis).

## Start here (reading order)

1. [papers/READING_LIST.md](papers/READING_LIST.md) — curated papers with why each matters
2. [docs/WHY_IT_IS_PROBLEMATIC.md](docs/WHY_IT_IS_PROBLEMATIC.md) — short problem framing
3. [docs/RESEARCH_DIRECTIONS.md](docs/RESEARCH_DIRECTIONS.md) — pick one track and scope it

## Top-line recommendations

| Priority | Paper | Why |
|----------|-------|-----|
| Must-read | Chancellor & De Choudhury (2020), *npj Digital Medicine* | Foundational critique of methods and construct validity |
| Must-read | Chancellor et al. (2019), FAT* ethics taxonomy | Maps consent, stigma, misuse, and stakeholder harm |
| Must-read | Harrigian et al. (2020), Findings of EMNLP | Shows weak labels + poor cross-platform generalization |
| Strong | Nguyen et al. (2022), ACL | PHQ-9 grounding improves OOD generalization |
| Strong | AAAI 2026 meta-analysis (“AI in the Wild”) | Recent evidence: high benchmarks, weak clinical/ethical reporting |
| Strong | JBDS systematic review of ML biases (mental illness on social media) | Sampling, language, labeling, evaluation bias catalog |
| Clinical bridge | Phillips & Kupfer / bipolar–MDD differential literature | Real-world misdiagnosis stakes beyond social-media ML |

Full citations and links are in the reading list.

## Project layout

```
misdiagnoses/
  README.md
  papers/
    READING_LIST.md
    references.bib
  docs/
    WHY_IT_IS_PROBLEMATIC.md
    RESEARCH_DIRECTIONS.md
  notes/
    .gitkeep
```

Use `notes/` for reading notes, paper summaries, and meeting agendas.

## Ethics note

Work in this area involves sensitive content (depression, self-harm language in datasets). Prefer published aggregates and ethically released corpora; do not scrape personal accounts without clear IRB/ethics oversight, and do not treat model scores as clinical diagnoses.
