# compliance

## Overview
Maps mltk assertion results to regulatory and security frameworks, reports coverage gaps, and generates compliance reports. HTML templates live in `compliance/templates` (Jinja2 `.j2` files, no code).

## Interface
- `eu_ai_act.py`: `classify_risk`, `map_results_to_articles`, `find_gaps`
- `nist_ai_rmf.py`: `classify_tier`, `map_results_to_measures`, `find_gaps`, `assert_nist_rmf_coverage`
- `iso_42001.py`: `map_results_to_clauses`, `find_gaps`, `assert_iso_42001_coverage`
- `hipaa.py`: `map_results_to_rules`, `find_gaps`, `assert_hipaa_coverage`
- `sr_11_7.py`: `classify_compliance`, `map_results_to_sections`, `find_gaps`, `assert_sr_11_7_coverage`
- `owasp_llm.py`: `owasp_llm_scan`, `assert_owasp_coverage`, `generate_owasp_report`
- `fda.py`: `generate_fda_audit_trail`
- `custom.py`: `load_custom_framework`, `map_results_to_custom`, `find_custom_gaps`, `assert_custom_coverage`
- `drift.py`: `FRAMEWORK_VERSIONS`, `snapshot_coverage`, `write_coverage_baseline`, `load_coverage_baseline`, `assert_no_compliance_drift`
- Package-level aliases in `__init__.py`: `find_nist_rmf_gaps`, `find_iso_42001_gaps`, `find_hipaa_gaps`, `find_sr_11_7_gaps` (each framework's `find_gaps`); bare `find_gaps` is the EU AI Act one.
- `generator.py`: `generate_compliance_report`
- `pdf_export.py`: `export_compliance_pdf`

## Invariants
- Mapping functions are computed only from the supplied test results.
- `assert_*` helpers raise on coverage gaps or drift from the committed baseline.

## Side Effects
- Report/PDF export and baseline writers write files; baseline loaders read files.

## Verification
- Run: `python -m pytest tests/test_compliance -q`
