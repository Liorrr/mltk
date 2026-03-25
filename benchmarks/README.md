# mltk Benchmarks

## Running

    python benchmarks/bench_drift.py          # Drift method speed
    python benchmarks/bench_pii.py            # PII scanning speed
    python benchmarks/bench_vs_competitors.py # Assertion speed

## Methodology
- Each assertion runs 100 times on 10K-row DataFrames
- Reports mean time in milliseconds
- Run on: [your machine specs]
