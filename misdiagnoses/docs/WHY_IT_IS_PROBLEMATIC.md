# Why depression detection is problematic

Short framing for advisors and students. Not a literature review—see the reading list for citations.

## 1. The attractive story

Social media + transformers → “detect depression early.” Papers report high accuracy/F1. Funders and press like screening-at-scale narratives.

## 2. What the labels usually are

Most public datasets do **not** use structured clinical interviews. Common proxies:

- Membership in depression-related subreddits
- Self-disclosure keywords (“I was diagnosed with…”)
- Crowd annotations of posts
- Occasionally PHQ-9 / CES-D scores (better, still not a full diagnosis)

Proxy labels correlate with *talking about* depression, community identity, or help-seeking language—not necessarily with a current Major Depressive Disorder episode.

**Misdiagnosis risk (computational):** the model learns “sounds like Reddit depression discourse” rather than “meets diagnostic criteria.”

## 3. What accuracy hides

Reported metrics can look excellent while systems fail on:

| Failure mode | What goes wrong |
|--------------|-----------------|
| Cross-platform transfer | Twitter ≠ Reddit ≠ clinical notes |
| Temporal shift | Language and platform norms change |
| Confounds | Topic, demographics, disclosure style inflate scores |
| Class imbalance | Accuracy without proper metrics misleads |
| Symptom collapse | One binary label ignores PHQ-9 dimensions and differentials (bipolar, anxiety, grief) |

High in-domain F1 is compatible with unsafe deployment.

## 4. Clinical misdiagnosis is already hard

Even expert clinicians struggle with differentials—especially **unipolar depression vs bipolar II**, where patients often present only while depressed and hypomania is under-reported. Wrong labels change treatment (antidepressants alone vs mood stabilizers) and outcomes.

Automated detectors that output a flat “depressed” flag can **amplify** that collapse: they do not reason about course, hypomania history, substances, medical mimics, or cultural expression of distress.

## 5. Harm is asymmetric

- **False positive:** stigma, unnecessary intervention, surveillance, insurance/employment misuse if systems leak outside care
- **False negative:** missed support when someone is struggling
- **Opaque models:** clinicians cannot justify decisions; patients cannot contest them
- **Consent gaps:** most users never opted into mental-health inference on public posts

Ethics papers argue these are not edge cases—they are structural.

## 6. What “good research” looks like here

Prefer work that:

1. Treats **construct validity** as the primary claim, not accuracy
2. Reports label provenance honestly
3. Tests **transfer** and temporal stability
4. Aligns outputs with clinical constructs (e.g., PHQ-9 symptoms) without claiming diagnosis
5. Measures annotation agreement with mental-health professionals
6. Discusses misuse, consent, and who is excluded by English/Twitter-heavy data

## One-sentence thesis for a student

> Automated depression detection often optimizes for weak social-media proxies and platform artifacts; without clinically grounded labels, evaluation, and safeguards, high benchmark scores can encode a form of systematic misdiagnosis.
