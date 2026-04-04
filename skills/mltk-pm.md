---
description: >
  mltk PM persona — interpreting ML Test Scores, compliance status,
  stakeholder reporting, and risk assessment for ML systems.
---

# mltk PM Skill — Product Manager Guide

## Role Summary

As a PM using mltk, your job is to interpret quality signals from ML scans, translate findings into business risk, drive prioritization, and communicate status to stakeholders. You do not write code — you read reports, ask developers to run commands, and make go/no-go decisions based on what the results say.

---

## ML Test Score

Run `mltk score` to get the aggregate ML Test Score for a model or dataset.

| Range | Status | Action |
|-------|--------|--------|
| 0–40 | Critical | Block deployment. Escalate immediately. |
| 40–70 | Needs work | Do not ship to production. Fix in current sprint. |
| 70–90 | Good | Ship with documented caveats. Track improvements. |
| 90–100 | Excellent | Production-ready. Monitor for drift. |

**Score components:** data quality, model performance, drift, fairness, safety. Ask the developer which component is pulling the score down — each maps to a specific scanner.

**Stakeholder communication:** "Our current ML Test Score is 74/100. Data quality and fairness are blocking us from hitting 90. We've prioritized those in this sprint."

---

## Reading Scan Reports

Run `mltk scan` (or ask a developer to) and interpret findings by severity:

| Severity | Meaning | Action |
|----------|---------|--------|
| CRITICAL | Production blocker — measurable risk of harm, failure, or legal exposure | Must fix before deploy |
| WARNING | Quality concern — degraded reliability or partial compliance gap | Fix in next sprint |
| INFO | Improvement opportunity — best practice not followed | Add to backlog |

**Prioritization formula:** Severity x Impact x (1 / Fix Effort)

- High severity + high user impact + low fix effort = do it now
- Low severity + low impact + high fix effort = defer or drop

Ask the developer to run `mltk scan --format json` and share the output. You need the `severity`, `scanner`, and `message` fields to prioritize.

---

## Compliance Dashboard

Use these commands to check compliance status per regulatory framework. Ask a developer to run them and share results.

| Framework | When to Use | CLI Command |
|-----------|-------------|-------------|
| FDA (SaMD) | Medical device software | `mltk fda-audit` |
| NIST AI RMF | US government, federal contracts | `mltk compliance --framework nist` |
| ISO 42001 | International AI management systems | `mltk compliance --framework iso42001` |
| EU AI Act | European market deployment | `mltk compliance --framework eu-ai-act` |
| OWASP LLM | LLM/agent applications | `mltk compliance --framework owasp-llm` |
| SR 11-7 | Banking and financial services | `mltk compliance --framework sr-11-7` |
| HIPAA | Healthcare data processing | `mltk compliance --framework hipaa` |

Each command outputs a pass/fail per control with a coverage percentage. A coverage below 80% is a compliance gap. Below 60% is a blocker for regulated markets.

---

## Stakeholder Report Generation

Three report output paths:

1. **AI-assisted report** — Use the `mltk_report` MCP tool. Generates a narrative summary of findings with recommended actions. Best for exec briefings.

2. **PDF compliance report** — Run `mltk compliance-pdf --framework <name>`. Produces a formatted document suitable for auditors and regulators.

3. **Dashboard metrics** — Run `mltk grafana-export` to push current metrics to Grafana. Use for ongoing tracking in team dashboards.

A complete report contains:
- Score summary (current vs. previous)
- Findings by severity (count and key examples)
- Trend over last N runs
- Compliance status per active framework
- Recommended next actions

---

## Risk Assessment Framework

When a scan returns findings, map them to business risk:

| Scanner Finding | Risk Type | PM Action |
|-----------------|-----------|-----------|
| Bias detected | Legal / reputational | Prioritize — involves protected groups or regulatory exposure |
| Drift detected | Model degradation | Decide: monitor (low drift) or retrain (high drift) |
| Data leakage | Data quality / trust | Fix pipeline before next training run |
| Calibration issue | Reliability / trust | Recalibrate before customer-facing deployment |
| LLM safety finding | Safety / brand | Immediate action — block deployment |
| Overfit detected | Generalization risk | Do not deploy to new population |
| Robustness failure | Reliability | Assess production input distribution; may block deploy |

Decision rule: any CRITICAL finding in bias, safety, or leakage categories is a hard blocker regardless of overall score.

---

## Key Metrics for Stakeholders

Track these in your sprint reviews and leadership updates:

- **ML Test Score trend** — week-over-week. Should trend up toward 90+.
- **CRITICAL finding count** — must be 0 for production models.
- **Compliance coverage %** — per active framework; target 80%+ before regulated launch.
- **Drift detection frequency** — how often drift alerts fire in production.
- **Model regression incidents** — deployments that degraded metrics post-launch.
- **Fairness metrics** — disparity ratios across protected groups (ask developer for `mltk scan --scanner bias` output).
- **Time to fix CRITICAL findings** — SLA metric for your team.

---

## Integration Points

mltk connects directly into PM-facing workflows:

- **GitHub Issues** — `mltk_create_issue` (MCP tool) converts a scan finding into a GitHub issue automatically. Use this to hand off CRITICAL/WARNING findings to the dev team with full context.
- **Jira** — `mltk_create_issue --backend jira` creates a Jira ticket instead. Assign to the responsible squad.
- **Slack** — `mltk notify slack` sends a scan summary to your team channel. Set up after each CI run for visibility.
- **CI/CD quality gates** — mltk can block deployments when score falls below a threshold or CRITICAL findings exist. Ask the infra team to configure the threshold. Typical gates: score >= 70, CRITICAL count == 0.

These integrations mean you do not need to manually read scan output to stay informed — set up notifications and review the tickets as they arrive.
