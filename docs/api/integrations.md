# Integrations (Jira / PM Tools)

Auto-create tickets from test failures. Supports Jira Cloud/Server with deduplication, severity filtering, and ML-specific ticket templates.

**Module:** `mltk.integrations`

---

## IssueTrackerAdapter
Abstract base class for PM tool integration. Implement for Jira, Linear, GitHub Issues, etc.

## JiraAdapter
Jira Cloud/Server implementation. Requires `jira` library (`pip install jira`).

## TicketDecisionEngine
Deduplication + spam prevention: content hash, cooldown, severity threshold.

## Ticket Templates
Pre-built templates for: data quality failure, model regression, drift detection, bias violation.

---
