# Alex Loftus — candidate profile + scoring rubric for the /jobs board

This file is the rating agent's ground truth. CV-level public facts only —
personal interview material lives in prep_bank.enc.json (encrypted).

## WHAT HE WANTS (most important — weight heavily)

A role where he can (1) LEAD PEOPLE, (2) give PRESENTATIONS to stakeholders,
(3) be SOCIAL and interact with people, while (4) staying HIGHLY TECHNICAL,
anchored in his interpretability/ML background. People-facing technical
leadership — "the room, not the repo." He is deliberately pivoting from
academia toward technical program/people leadership, but does NOT want to
leave technical depth behind.

## CURRENT ROLE

Red-Teaming Program Lead, OpenAI (contract via Handshake), Mar 2026–present.
Designed & led TWO multi-agent LLM red-teaming campaigns end to end: wrote the
proposals, recruited & led a 16-person team (doubled for 2nd campaign), built
~17K lines of evaluation/observability infrastructure in under two weeks, ran
live 24/7 operations + incident response, and translated emergent risk signals
into structured findings/recommendations for OpenAI stakeholders.

## RESEARCH

PhD student (2024–present), Northeastern, advisor David Bau. Mechanistic
interpretability, evaluation of LLMs, data attribution (~2 yrs LLM interp/eval;
8 yrs ML total). Co-first author NNsight/NDIF (ICLR 2025, open interpretability
infrastructure). Author on "Agents of Chaos" (covered by Science, WIRED, Jack
Clark's Import AI). Subliminal-learning token-entanglement (NeurIPS workshop
2025).

## COMMUNICATION / LEADERSHIP EVIDENCE

Textbook author (Cambridge UP 2025, 524pp); linear-algebra YouTube tutoring
series; 10+ invited talks (20–300 people, incl. DC policymakers and LLNL);
co-organized 200-person NEMI interpretability conference + ran day-of
operations; Vesuvius Kaggle 1st place ($100K, 1,249 teams, Scientific American
cover); taught weekly comp-neuro seminars; founded comp-neuro club. EleutherAI
Head of Growth (funding strategy 2025); Harvard AI Safety Team
technical-lead/facilitator (2026); CBAI research mentor; Krnel.ai strategic
advisor.

## PAST ROLES

Data Scientist, Creyon Bio (protein-model eval/benchmarking, 2023–24); ML
Research Engineer, BlueHalo (diffusion synthetic data, DARPA demos, 2022–23);
Research SWE, JHU NeuroData/Vogelstein (MRI pipeline on AWS Batch/Docker,
2018–20); Assistant Director, iD Tech Camps (managed 10+ instructors, 300+
students, curriculum to 50+ locations, 2014–18).

## SKILLS

Python (advanced), eval frameworks (Inspect AI, lm-eval-harness), red-teaming,
harm-taxonomy design, interpretability (NNsight), Fly.io/Docker/AWS (Batch,
EKS/K8s, SSO)/FastAPI/GCP/Firebase, monitoring dashboards, a personal
GPU-scheduler. Languages: Python, Bash, JS, SQL.

## EDUCATION

MSE Biomedical Eng (JHU, ML/DS focus, GPA 3.97); BS Behavioral Neuroscience
(WWU, minors chem + philosophy).

## HARD CONSTRAINTS / REALITIES (use for calibration — do not inflate)

- US-based (Boston, PhD at Northeastern) and WILLING TO RELOCATE (SF/NYC
  hybrid is fine) for the right role — do NOT penalize US locations. Non-US-only
  roles (Tokyo, Sydney, London-only, Dublin, Seoul, Munich, Paris, Bangalore,
  Singapore) remain a major negative unless explicitly remote-US-friendly.
- He has NO formal full-time industry TPM or people-management title — his
  leadership is via a contract lead role, research orgs, and volunteer/community
  orgs. Roles requiring "8+ yrs TPM" or "5+ yrs managing engineers as an FTE
  manager" are a stretch.
- PhD is in progress; he would take leave for a full-time role.
- He is NOT a staff/principal-level production software engineer (no big-tech
  FTE SWE tenure). Staff+/Senior-Staff pure-SWE roles (backend, mobile,
  databases, kernel, inference runtime, frontend) are poor fits despite strong
  coding.
- His deepest domains: AI safety, red-teaming, evaluation, interpretability,
  research-adjacent program leadership, technical communication.
- ANTHROPIC ONLY: he has a REFERRAL (a current Anthropic employee will submit
  an internal referral). This materially raises the chance of reaching a
  recruiter screen but does NOT lower the hiring bar. Other companies: no
  referral assumed.

## SCORING DIMENSIONS (integers 1–10 each)

- **people** — how much the role has him leading/recruiting/managing/mentoring
  humans. 10 = building and running a team; 1 = solo IC heads-down.
- **stage** — presentations, exec briefings, stakeholder communication, public
  or cross-org visibility. 10 = routinely presenting to leadership/external
  parties; 1 = never presents.
- **social** — daily human interaction density: cross-team coordination,
  partners, candidates, community. 10 = talking to people most of the day.
- **technical** — how technical the role stays, AND whether Alex clears its
  technical bar. 10 = deeply technical in ways he's strong at (evals,
  red-teaming, interp, ML); low if either non-technical OR technical in ways
  he isn't (e.g. staff production SWE, silicon, security engineering).
- **domain** — overlap with his domains: AI safety, red-teaming, evals,
  interpretability, research programs, technical communication. 10 = dead
  center; 1 = unrelated (sales, finance, legal).

A deterministic script computes fit = .24·people + .20·stage + .12·social +
.22·technical + .22·domain. Score dimensions honestly; don't reverse-engineer
a desired fit.

## PROBABILITY CALIBRATION

prob = realistic probability (integer %) of an OFFER conditional on applying
(with the referral, for Anthropic). Frontier-lab bars are very high: even an
excellent fit is rarely above ~25–30%; solid-but-contested ~10–20%; a stretch
~3–8%; clear mismatch <3%.

Anchor points from the 2026-06-17 full ranking (hold consistency with these):
- Technical Program Management, Alignment (Anthropic) = 18
- Technical Program Manager, Research (Anthropic) = 16
- Research Operations Discovery-type roles = ~14
- TPM Safeguards Infra & Evals (Anthropic) = 13 (SRE/on-call core — "repo not
  room" penalty on role-shape, not on ability)
- Staff+ pure-SWE roles = 1–2; Sales/GTM/Finance = 0–2.
