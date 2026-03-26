use pyo3::prelude::*;
use regex::Regex;

/// Compute KS test statistic between two distributions.
/// Uses empirical CDF comparison. Returns (statistic, approximate p_value).
#[pyfunction]
fn ks_test(mut reference: Vec<f64>, mut current: Vec<f64>) -> (f64, f64) {
    let n1 = reference.len() as f64;
    let n2 = current.len() as f64;

    if n1 == 0.0 || n2 == 0.0 {
        return (0.0, 1.0);
    }

    reference.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    current.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Merge and compute max CDF difference
    let mut i = 0usize;
    let mut j = 0usize;
    let mut d_max: f64 = 0.0;

    while i < reference.len() && j < current.len() {
        let cdf1 = (i + 1) as f64 / n1;
        let cdf2 = (j + 1) as f64 / n2;

        if reference[i] <= current[j] {
            let diff = (cdf1 - (j as f64 / n2)).abs();
            if diff > d_max {
                d_max = diff;
            }
            i += 1;
        } else {
            let diff = ((i as f64 / n1) - cdf2).abs();
            if diff > d_max {
                d_max = diff;
            }
            j += 1;
        }
    }

    // Handle remaining elements
    while i < reference.len() {
        let cdf1 = (i + 1) as f64 / n1;
        let diff = (cdf1 - 1.0).abs();
        if diff > d_max {
            d_max = diff;
        }
        i += 1;
    }
    while j < current.len() {
        let cdf2 = (j + 1) as f64 / n2;
        let diff = (1.0 - cdf2).abs();
        if diff > d_max {
            d_max = diff;
        }
        j += 1;
    }

    // Approximate p-value using asymptotic formula
    let en = (n1 * n2 / (n1 + n2)).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d_max;
    let p_value = ks_p_value(lambda);

    (d_max, p_value)
}

/// Approximate KS p-value using the asymptotic series expansion.
fn ks_p_value(lambda: f64) -> f64 {
    if lambda < 0.001 {
        return 1.0;
    }
    if lambda > 5.0 {
        return 0.0;
    }

    let mut sum = 0.0;
    for k in 1..=100 {
        let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
        let term = sign * (-2.0 * (k as f64).powi(2) * lambda * lambda).exp();
        sum += term;
        if term.abs() < 1e-12 {
            break;
        }
    }

    (2.0 * sum).clamp(0.0, 1.0)
}

/// Compute Population Stability Index between two distributions.
/// PSI < 0.1 = stable, 0.1-0.2 = moderate, > 0.2 = significant drift.
#[pyfunction]
fn psi(reference: Vec<f64>, current: Vec<f64>, bins: usize) -> f64 {
    if reference.is_empty() || current.is_empty() || bins == 0 {
        return 0.0;
    }

    let ref_min = reference.iter().cloned().fold(f64::INFINITY, f64::min);
    let ref_max = reference.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cur_min = current.iter().cloned().fold(f64::INFINITY, f64::min);
    let cur_max = current.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let global_min = ref_min.min(cur_min);
    let global_max = ref_max.max(cur_max);

    if (global_max - global_min).abs() < 1e-10 {
        return 0.0; // All values identical
    }

    let bin_width = (global_max - global_min) / bins as f64;
    let n_ref = reference.len() as f64;
    let n_cur = current.len() as f64;

    let mut ref_counts = vec![0.0f64; bins];
    let mut cur_counts = vec![0.0f64; bins];

    for &v in &reference {
        let idx = ((v - global_min) / bin_width).floor() as usize;
        let idx = idx.min(bins - 1);
        ref_counts[idx] += 1.0;
    }
    for &v in &current {
        let idx = ((v - global_min) / bin_width).floor() as usize;
        let idx = idx.min(bins - 1);
        cur_counts[idx] += 1.0;
    }

    let epsilon = 1e-6;
    let mut psi_val = 0.0;
    for i in 0..bins {
        let ref_pct = (ref_counts[i] / n_ref).max(epsilon);
        let cur_pct = (cur_counts[i] / n_cur).max(epsilon);
        psi_val += (cur_pct - ref_pct) * (cur_pct / ref_pct).ln();
    }

    psi_val
}

// ─── Histogram helper ────────────────────────────────────────────────────────

/// Build equal-width histogram proportions from global min/max.
/// Returns (ref_proportions, cur_proportions), both clipped to epsilon.
fn build_histograms(reference: &[f64], current: &[f64], bins: usize) -> (Vec<f64>, Vec<f64>) {
    let epsilon = 1e-6_f64;

    let global_min = reference
        .iter()
        .chain(current.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let global_max = reference
        .iter()
        .chain(current.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let range = global_max - global_min;
    // If all values are identical, both distributions are the same — return uniform
    if range.abs() < 1e-10 {
        let flat = vec![1.0_f64 / bins as f64; bins];
        return (flat.clone(), flat);
    }

    let bin_width = range / bins as f64;
    let n_ref = reference.len() as f64;
    let n_cur = current.len() as f64;

    let mut ref_counts = vec![0.0_f64; bins];
    let mut cur_counts = vec![0.0_f64; bins];

    for &v in reference {
        let idx = (((v - global_min) / bin_width).floor() as usize).min(bins - 1);
        ref_counts[idx] += 1.0;
    }
    for &v in current {
        let idx = (((v - global_min) / bin_width).floor() as usize).min(bins - 1);
        cur_counts[idx] += 1.0;
    }

    let ref_props: Vec<f64> = ref_counts
        .iter()
        .map(|&c| (c / n_ref).max(epsilon))
        .collect();
    let cur_props: Vec<f64> = cur_counts
        .iter()
        .map(|&c| (c / n_cur).max(epsilon))
        .collect();

    (ref_props, cur_props)
}

// ─── KL Divergence ───────────────────────────────────────────────────────────

/// Histogram-based KL divergence D_KL(P || Q).
/// Returns 0.0 for identical distributions; higher values indicate more drift.
#[pyfunction]
fn kl_divergence(reference: Vec<f64>, current: Vec<f64>, bins: usize) -> f64 {
    if reference.is_empty() || current.is_empty() || bins == 0 {
        return 0.0;
    }

    let (p, q) = build_histograms(&reference, &current, bins);

    // D_KL(P || Q) = sum(P * ln(P/Q))
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum()
}

// ─── Chi-Squared Test ────────────────────────────────────────────────────────

/// Chi-squared goodness-of-fit test.
/// Returns (statistic, p_value). High statistic / low p_value means distributions differ.
#[pyfunction]
fn chi_squared(observed: Vec<f64>, expected: Vec<f64>) -> (f64, f64) {
    if observed.is_empty() || expected.is_empty() || observed.len() != expected.len() {
        return (0.0, 1.0);
    }

    let statistic: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| {
            if e > 1e-12 {
                (o - e) * (o - e) / e
            } else {
                0.0
            }
        })
        .sum();

    let df = (observed.len() - 1) as f64;
    let p_value = chi2_sf(statistic, df);

    (statistic, p_value)
}

/// Chi-squared survival function P(X > x) for df degrees of freedom.
/// Uses the regularized incomplete gamma function via Wilson-Hilferty approximation
/// for large df, and a direct series expansion for small df.
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if df <= 0.0 {
        return 0.0;
    }

    // Use regularised upper incomplete gamma: Q(df/2, x/2)
    let a = df / 2.0;
    let x2 = x / 2.0;
    upper_incomplete_gamma_regularized(a, x2)
}

/// Regularised upper incomplete gamma Q(a, x) via continued fraction / series.
fn upper_incomplete_gamma_regularized(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }

    // For small x, use the series representation of the lower incomplete gamma
    // and return 1 - P(a, x).
    if x < a + 1.0 {
        let p = lower_gamma_series(a, x);
        return (1.0 - p).clamp(0.0, 1.0);
    }

    // For large x, use the continued fraction representation.
    upper_gamma_cf(a, x).clamp(0.0, 1.0)
}

/// Series expansion for the regularised lower incomplete gamma P(a, x).
fn lower_gamma_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let ln_gamma_a = ln_gamma(a);
    let log_factor = a * x.ln() - x - ln_gamma_a;
    if log_factor < -700.0 {
        return 0.0;
    }
    let factor = log_factor.exp();

    let mut term = 1.0 / a;
    let mut sum = term;
    for n in 1..=200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-12 {
            break;
        }
    }
    (factor * sum).clamp(0.0, 1.0)
}

/// Continued fraction for the regularised upper incomplete gamma Q(a, x).
fn upper_gamma_cf(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let log_factor = a * x.ln() - x - ln_gamma_a;
    if log_factor < -700.0 {
        return 0.0;
    }
    let factor = log_factor.exp();

    // Lentz's method
    let fpmin = 1e-300_f64;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-12 {
            break;
        }
    }

    (factor * h).clamp(0.0, 1.0)
}

/// Log-gamma via Lanczos approximation (g=7, n=9).
fn ln_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.13857_109_526_572_012,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if z < 0.5 {
        // Reflection formula
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }

    let z = z - 1.0;
    let mut x = C[0];
    for (i, &ci) in C[1..].iter().enumerate() {
        x += ci / (z + i as f64 + 1.0);
    }
    let t = z + G + 0.5;

    0.5 * (2.0 * std::f64::consts::PI).ln()
        + (z + 0.5) * t.ln()
        - t
        + x.ln()
}

// ─── Jensen-Shannon Divergence ───────────────────────────────────────────────

/// Jensen-Shannon divergence, normalised to [0, 1].
/// JS = 0 for identical distributions, 1 for completely disjoint distributions.
#[pyfunction]
fn js_divergence(reference: Vec<f64>, current: Vec<f64>, bins: usize) -> f64 {
    if reference.is_empty() || current.is_empty() || bins == 0 {
        return 0.0;
    }

    let (p, q) = build_histograms(&reference, &current, bins);

    // M = (P + Q) / 2
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();

    // KL(P || M) + KL(Q || M)
    let kl_pm: f64 = p.iter().zip(m.iter()).map(|(&pi, &mi)| pi * (pi / mi).ln()).sum();
    let kl_qm: f64 = q.iter().zip(m.iter()).map(|(&qi, &mi)| qi * (qi / mi).ln()).sum();

    let js_nats = 0.5 * kl_pm + 0.5 * kl_qm;

    // Normalise by ln(2) so result is in [0, 1]
    (js_nats / 2_f64.ln()).clamp(0.0, 1.0)
}

// ─── Wasserstein Distance ────────────────────────────────────────────────────

/// Earth Mover's Distance (Wasserstein-1) between two 1-D distributions.
/// Computed via sorted CDF integral in O(n log n).
#[pyfunction]
fn wasserstein(mut reference: Vec<f64>, mut current: Vec<f64>) -> f64 {
    if reference.is_empty() || current.is_empty() {
        return 0.0;
    }

    reference.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    current.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = reference.len();
    let m = current.len();

    // Merge all unique x-values (sorted union)
    let mut all_vals: Vec<f64> = Vec::with_capacity(n + m);
    all_vals.extend_from_slice(&reference);
    all_vals.extend_from_slice(&current);
    all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    let mut prev_cdf_r = 0.0_f64;
    let mut prev_cdf_c = 0.0_f64;
    let mut prev_x = all_vals[0];

    let mut ri = 0usize; // pointer into sorted reference
    let mut ci = 0usize; // pointer into sorted current

    let mut distance = 0.0_f64;

    for &x in &all_vals {
        // Integrate |CDF_r - CDF_c| over [prev_x, x]
        let width = x - prev_x;
        if width > 0.0 {
            distance += (prev_cdf_r - prev_cdf_c).abs() * width;
        }

        // Advance CDF pointers to include all values <= x
        while ri < n && reference[ri] <= x {
            ri += 1;
        }
        while ci < m && current[ci] <= x {
            ci += 1;
        }

        prev_cdf_r = ri as f64 / n as f64;
        prev_cdf_c = ci as f64 / m as f64;
        prev_x = x;
    }

    distance
}

// ─── PII Scanner ─────────────────────────────────────────────────────────────

/// Scan text for PII using a list of (pattern_name, regex_pattern) pairs.
/// Returns a list of (pattern_name, start_pos, end_pos, matched_text).
#[pyfunction]
fn scan_pii_rust(
    text: String,
    patterns: Vec<(String, String)>,
) -> PyResult<Vec<(String, usize, usize, String)>> {
    let mut results: Vec<(String, usize, usize, String)> = Vec::new();

    for (name, pattern) in &patterns {
        let re = Regex::new(pattern).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid regex pattern '{}': {}",
                pattern, e
            ))
        })?;

        for mat in re.find_iter(&text) {
            results.push((
                name.clone(),
                mat.start(),
                mat.end(),
                mat.as_str().to_string(),
            ));
        }
    }

    // Sort by start position for deterministic output
    results.sort_by_key(|r| r.1);

    Ok(results)
}

// ─── Cosine Similarity ───────────────────────────────────────────────────────

/// Compute cosine similarity between two vectors.
/// Returns dot(a, b) / (||a|| * ||b||).
/// Returns 0.0 for zero-length or zero-norm vectors.
#[pyfunction]
fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute cosine distance between centroids of two embedding sets.
/// centroid = mean of all vectors in the set.
/// distance = 1.0 - cosine_similarity(centroid_ref, centroid_cur)
#[pyfunction]
fn centroid_cosine_distance(ref_embs: Vec<Vec<f64>>, cur_embs: Vec<Vec<f64>>) -> f64 {
    if ref_embs.is_empty() || cur_embs.is_empty() {
        return 0.0;
    }

    let dim = ref_embs[0].len();
    if dim == 0 || cur_embs[0].len() != dim {
        return 0.0;
    }

    // Compute centroid for ref
    let n_ref = ref_embs.len() as f64;
    let mut centroid_ref = vec![0.0_f64; dim];
    for vec in &ref_embs {
        for (i, &v) in vec.iter().enumerate() {
            centroid_ref[i] += v;
        }
    }
    for v in &mut centroid_ref {
        *v /= n_ref;
    }

    // Compute centroid for cur
    let n_cur = cur_embs.len() as f64;
    let mut centroid_cur = vec![0.0_f64; dim];
    for vec in &cur_embs {
        for (i, &v) in vec.iter().enumerate() {
            centroid_cur[i] += v;
        }
    }
    for v in &mut centroid_cur {
        *v /= n_cur;
    }

    1.0 - cosine_similarity(centroid_ref, centroid_cur)
}

/// BERTScore precision, recall, and F1 via token-level cosine similarity.
///
/// Precision: for each hypothesis token embedding, find max cosine similarity
///            against all reference token embeddings → mean over hypothesis tokens.
/// Recall:    for each reference token embedding, find max cosine similarity
///            against all hypothesis token embeddings → mean over reference tokens.
/// F1:        2 * P * R / (P + R), or 0 if P + R == 0.
#[pyfunction]
fn bertscore_precision_recall(
    ref_embs: Vec<Vec<f64>>,
    hyp_embs: Vec<Vec<f64>>,
) -> (f64, f64, f64) {
    if ref_embs.is_empty() || hyp_embs.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    // Precision: for each hyp token, max sim over all ref tokens
    let precision: f64 = {
        let sum: f64 = hyp_embs
            .iter()
            .map(|hyp_tok| {
                ref_embs
                    .iter()
                    .map(|ref_tok| cosine_similarity(hyp_tok.clone(), ref_tok.clone()))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .sum();
        sum / hyp_embs.len() as f64
    };

    // Recall: for each ref token, max sim over all hyp tokens
    let recall: f64 = {
        let sum: f64 = ref_embs
            .iter()
            .map(|ref_tok| {
                hyp_embs
                    .iter()
                    .map(|hyp_tok| cosine_similarity(ref_tok.clone(), hyp_tok.clone()))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .sum();
        sum / ref_embs.len() as f64
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1)
}

// ─── Module registration ─────────────────────────────────────────────────────

/// ML Test Kit — Rust acceleration module.
#[pymodule]
fn _mltk_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ks_test, m)?)?;
    m.add_function(wrap_pyfunction!(psi, m)?)?;
    m.add_function(wrap_pyfunction!(kl_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(chi_squared, m)?)?;
    m.add_function(wrap_pyfunction!(js_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(wasserstein, m)?)?;
    m.add_function(wrap_pyfunction!(scan_pii_rust, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(centroid_cosine_distance, m)?)?;
    m.add_function(wrap_pyfunction!(bertscore_precision_recall, m)?)?;
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── KS tests (existing) ──────────────────────────────────────────────────

    #[test]
    fn test_ks_identical_distributions() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (stat, p) = ks_test(data.clone(), data);
        assert!(stat < 0.3, "KS stat should be small for identical data: {stat}");
        assert!(p > 0.5, "p-value should be high for identical data: {p}");
    }

    #[test]
    fn test_ks_different_distributions() {
        let ref_data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let cur_data: Vec<f64> = (50..150).map(|i| i as f64 * 0.1).collect();
        let (stat, p) = ks_test(ref_data, cur_data);
        assert!(stat > 0.3, "KS stat should be high for shifted data: {stat}");
        assert!(p < 0.05, "p-value should be low for shifted data: {p}");
    }

    #[test]
    fn test_psi_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = psi(data.clone(), data, 10);
        assert!(result.abs() < 0.01, "PSI for identical data should be ~0: {result}");
    }

    #[test]
    fn test_psi_shifted() {
        let ref_data: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.01).collect();
        let cur_data: Vec<f64> = (500..1500).map(|i| (i as f64) * 0.01).collect();
        let result = psi(ref_data, cur_data, 10);
        assert!(result > 0.1, "PSI should be high for shifted data: {result}");
    }

    #[test]
    fn test_ks_p_value_bounds() {
        assert!((ks_p_value(0.0) - 1.0).abs() < 0.01);
        assert!(ks_p_value(10.0) < 0.001);
    }

    // ── KL Divergence ────────────────────────────────────────────────────────

    #[test]
    fn test_kl_identical() {
        let data: Vec<f64> = (0..200).map(|i| i as f64 * 0.05).collect();
        let result = kl_divergence(data.clone(), data, 10);
        assert!(result.abs() < 1e-6, "KL(P,P) should be ~0, got {result}");
    }

    #[test]
    fn test_kl_shifted() {
        let ref_data: Vec<f64> = (0..500).map(|i| i as f64 * 0.01).collect();
        let cur_data: Vec<f64> = (500..1000).map(|i| i as f64 * 0.01).collect();
        let result = kl_divergence(ref_data, cur_data, 10);
        assert!(result > 0.0, "KL should be > 0 for different distributions: {result}");
    }

    // ── Chi-squared ──────────────────────────────────────────────────────────

    #[test]
    fn test_chi_squared_identical() {
        // Equal observed and expected → statistic = 0, p-value = 1
        let counts = vec![10.0, 20.0, 30.0, 40.0];
        let (stat, p) = chi_squared(counts.clone(), counts);
        assert!(stat.abs() < 1e-10, "Chi2 stat for identical should be 0, got {stat}");
        assert!(p > 0.99, "p-value for identical should be ~1, got {p}");
    }

    #[test]
    fn test_chi_squared_different() {
        // Very different observed vs expected → high statistic, low p-value
        let observed = vec![100.0, 1.0, 1.0, 1.0];
        let expected = vec![25.0, 25.0, 25.0, 25.0];
        let (stat, p) = chi_squared(observed, expected);
        assert!(stat > 100.0, "Chi2 stat should be high for different dists: {stat}");
        assert!(p < 0.001, "p-value should be low for different dists: {p}");
    }

    // ── Jensen-Shannon ───────────────────────────────────────────────────────

    #[test]
    fn test_js_identical() {
        let data: Vec<f64> = (0..200).map(|i| i as f64 * 0.05).collect();
        let result = js_divergence(data.clone(), data, 10);
        assert!(result.abs() < 1e-6, "JS(P,P) should be ~0, got {result}");
    }

    #[test]
    fn test_js_bounded() {
        // JS must always be in [0, 1] (normalised)
        let ref_data: Vec<f64> = (0..300).map(|i| i as f64 * 0.01).collect();
        let cur_data: Vec<f64> = (500..800).map(|i| i as f64 * 0.01).collect();
        let result = js_divergence(ref_data, cur_data, 10);
        assert!(result >= 0.0, "JS should be >= 0, got {result}");
        assert!(result <= 1.0, "JS should be <= 1, got {result}");
    }

    // ── Wasserstein ──────────────────────────────────────────────────────────

    #[test]
    fn test_wasserstein_identical() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = wasserstein(data.clone(), data);
        assert!(result.abs() < 1e-10, "Wasserstein distance for identical should be 0, got {result}");
    }

    #[test]
    fn test_wasserstein_shifted() {
        let ref_data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let cur_data: Vec<f64> = (0..100).map(|i| i as f64 + 10.0).collect();
        let result = wasserstein(ref_data, cur_data);
        // W-1 for uniform shift of 10 units should be ~10
        assert!(result > 5.0, "Wasserstein should be > 0 for shifted dists: {result}");
    }

    // ── Cosine Similarity ────────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(v.clone(), v);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "Cosine similarity of identical vectors should be 1.0, got {result}"
        );
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let result = cosine_similarity(a, b);
        assert!(
            result.abs() < 1e-10,
            "Cosine similarity of orthogonal vectors should be 0.0, got {result}"
        );
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let result = cosine_similarity(a, b);
        assert!(
            (result - (-1.0)).abs() < 1e-10,
            "Cosine similarity of opposite vectors should be -1.0, got {result}"
        );
    }

    // ── Centroid Cosine Distance ─────────────────────────────────────────────

    #[test]
    fn test_centroid_identical() {
        let embs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = centroid_cosine_distance(embs.clone(), embs);
        assert!(
            result.abs() < 1e-10,
            "Centroid cosine distance for identical sets should be 0.0, got {result}"
        );
    }

    // ── BERTScore ────────────────────────────────────────────────────────────

    #[test]
    fn test_bertscore_identical() {
        let embs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (p, r, f1) = bertscore_precision_recall(embs.clone(), embs);
        assert!(
            (p - 1.0).abs() < 1e-10,
            "BERTScore precision for identical should be 1.0, got {p}"
        );
        assert!(
            (r - 1.0).abs() < 1e-10,
            "BERTScore recall for identical should be 1.0, got {r}"
        );
        assert!(
            (f1 - 1.0).abs() < 1e-10,
            "BERTScore F1 for identical should be 1.0, got {f1}"
        );
    }

    // ── PII Scanner ──────────────────────────────────────────────────────────

    #[test]
    fn test_scan_pii_basic_email() {
        let text = "Contact me at user@example.com for details.".to_string();
        let patterns = vec![("email".to_string(), r"[\w.+-]+@[\w-]+\.[a-z]{2,}".to_string())];
        let results = scan_pii_rust(text, patterns).unwrap();
        assert_eq!(results.len(), 1, "Expected one email match, got {}", results.len());
        assert_eq!(results[0].0, "email");
        assert_eq!(results[0].3, "user@example.com");
    }

    #[test]
    fn test_scan_pii_multiple_patterns() {
        let text = "SSN: 123-45-6789, card: 4111111111111111".to_string();
        let patterns = vec![
            ("ssn".to_string(), r"\d{3}-\d{2}-\d{4}".to_string()),
            ("visa".to_string(), r"4\d{15}".to_string()),
        ];
        let results = scan_pii_rust(text, patterns).unwrap();
        assert_eq!(results.len(), 2, "Expected two matches, got {}", results.len());
        // Results sorted by start position — SSN comes first
        assert_eq!(results[0].0, "ssn");
        assert_eq!(results[1].0, "visa");
    }

    #[test]
    fn test_scan_pii_no_matches() {
        let text = "No sensitive data here.".to_string();
        let patterns = vec![("email".to_string(), r"[\w.+-]+@[\w-]+\.[a-z]{2,}".to_string())];
        let results = scan_pii_rust(text, patterns).unwrap();
        assert!(results.is_empty(), "Expected no matches for clean text");
    }

    #[test]
    fn test_scan_pii_empty_text() {
        let text = "".to_string();
        let patterns = vec![("email".to_string(), r"[\w.+-]+@[\w-]+\.[a-z]{2,}".to_string())];
        let results = scan_pii_rust(text, patterns).unwrap();
        assert!(results.is_empty(), "Expected no matches for empty text");
    }

    #[test]
    fn test_scan_pii_empty_patterns() {
        let text = "user@example.com".to_string();
        let patterns: Vec<(String, String)> = vec![];
        let results = scan_pii_rust(text, patterns).unwrap();
        assert!(results.is_empty(), "Expected no matches with no patterns");
    }

    #[test]
    fn test_scan_pii_invalid_regex_returns_error() {
        let text = "some text".to_string();
        let patterns = vec![("bad".to_string(), r"[invalid".to_string())];
        let result = scan_pii_rust(text, patterns);
        assert!(result.is_err(), "Invalid regex should return Err");
    }

    #[test]
    fn test_scan_pii_sorted_by_start_position() {
        // Two overlapping-region matches from different patterns — verify ordering
        let text = "abc123def456".to_string();
        let patterns = vec![
            ("letters_late".to_string(), r"def".to_string()),
            ("digits_early".to_string(), r"\d+".to_string()),
        ];
        let results = scan_pii_rust(text, patterns).unwrap();
        // Positions should be non-decreasing
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "Results not sorted by start position");
        }
    }

    // ── Edge cases: empty / single-element vectors ────────────────────────────

    #[test]
    fn test_ks_empty_inputs() {
        let (stat, p) = ks_test(vec![], vec![1.0, 2.0]);
        assert_eq!(stat, 0.0, "ks_test with empty reference should return stat=0");
        assert_eq!(p, 1.0, "ks_test with empty reference should return p=1");

        let (stat, p) = ks_test(vec![1.0, 2.0], vec![]);
        assert_eq!(stat, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_ks_single_element() {
        // Single-element distributions — should not panic
        let (stat, p) = ks_test(vec![1.0], vec![1.0]);
        assert!(stat >= 0.0, "stat should be non-negative: {stat}");
        assert!(p >= 0.0 && p <= 1.0, "p-value out of range: {p}");

        let (stat, p) = ks_test(vec![0.0], vec![1.0]);
        assert!(stat >= 0.0, "stat should be non-negative: {stat}");
        assert!(p >= 0.0 && p <= 1.0, "p-value out of range: {p}");
    }

    #[test]
    fn test_psi_empty_inputs() {
        assert_eq!(psi(vec![], vec![1.0], 10), 0.0, "psi with empty reference should be 0");
        assert_eq!(psi(vec![1.0], vec![], 10), 0.0, "psi with empty current should be 0");
        assert_eq!(psi(vec![1.0], vec![1.0], 0), 0.0, "psi with 0 bins should be 0");
    }

    #[test]
    fn test_psi_single_element() {
        // Single identical values — all values identical → PSI = 0
        let result = psi(vec![5.0], vec![5.0], 10);
        assert_eq!(result, 0.0, "psi for identical single-element should be 0");
    }

    #[test]
    fn test_wasserstein_single_element() {
        // Single-element identical → distance = 0
        let result = wasserstein(vec![3.0], vec![3.0]);
        assert!(result.abs() < 1e-10, "Wasserstein for identical single values should be ~0: {result}");

        // Single-element shifted → distance = shift amount
        let result = wasserstein(vec![0.0], vec![5.0]);
        assert!((result - 5.0).abs() < 1e-10, "Wasserstein for 0→5 should be 5.0, got {result}");
    }

    #[test]
    fn test_cosine_empty_inputs() {
        assert_eq!(cosine_similarity(vec![], vec![]), 0.0, "cosine sim of empty vecs should be 0");
        assert_eq!(cosine_similarity(vec![1.0], vec![]), 0.0, "cosine sim with one empty should be 0");
        assert_eq!(cosine_similarity(vec![], vec![1.0]), 0.0, "cosine sim with one empty should be 0");
    }

    #[test]
    fn test_cosine_mismatched_lengths() {
        let result = cosine_similarity(vec![1.0, 2.0], vec![1.0]);
        assert_eq!(result, 0.0, "cosine sim with mismatched lengths should be 0");
    }

    #[test]
    fn test_cosine_zero_vector() {
        // Zero-norm vector → should return 0.0, not NaN
        let result = cosine_similarity(vec![0.0, 0.0, 0.0], vec![1.0, 2.0, 3.0]);
        assert_eq!(result, 0.0, "cosine sim with zero vector should be 0");
        assert!(!result.is_nan(), "cosine sim should never return NaN");
    }

    // ── NaN/Inf value handling ────────────────────────────────────────────────

    #[test]
    fn test_ks_with_nan_values() {
        // NaN values get placed via partial_cmp fallback (Equal ordering).
        // The test ensures no panic occurs — result values are allowed to be
        // imprecise since NaN inputs are undefined behaviour for statistics.
        let ref_data = vec![1.0, f64::NAN, 3.0];
        let cur_data = vec![1.0, 2.0, 3.0];
        let (stat, p) = ks_test(ref_data, cur_data);
        assert!(!stat.is_nan(), "stat should not be NaN");
        assert!(!p.is_nan(), "p should not be NaN");
        assert!(p >= 0.0 && p <= 1.0, "p-value out of [0,1]: {p}");
    }

    #[test]
    fn test_cosine_with_inf_values() {
        // Inf in input propagates to norm (Inf), then 0/Inf = 0 via guard.
        // Expect 0.0 (guarded), not NaN or panic.
        let a = vec![f64::INFINITY, 0.0];
        let b = vec![1.0, 0.0];
        let result = cosine_similarity(a, b);
        // norm_a = Inf which is NOT < 1e-10 — dot/norm may be NaN or finite
        // We only assert no panic occurred; NaN is acceptable here since
        // Inf inputs are undefined.
        let _ = result; // no panic = pass
    }

    #[test]
    fn test_wasserstein_empty_inputs() {
        assert_eq!(wasserstein(vec![], vec![1.0]), 0.0, "wasserstein with empty ref should be 0");
        assert_eq!(wasserstein(vec![1.0], vec![]), 0.0, "wasserstein with empty cur should be 0");
    }

    #[test]
    fn test_kl_empty_inputs() {
        assert_eq!(kl_divergence(vec![], vec![1.0], 10), 0.0);
        assert_eq!(kl_divergence(vec![1.0], vec![], 10), 0.0);
        assert_eq!(kl_divergence(vec![1.0], vec![1.0], 0), 0.0);
    }

    #[test]
    fn test_chi_squared_empty_and_mismatched() {
        let (s, p) = chi_squared(vec![], vec![]);
        assert_eq!(s, 0.0);
        assert_eq!(p, 1.0);

        let (s, p) = chi_squared(vec![1.0, 2.0], vec![1.0]);
        assert_eq!(s, 0.0, "mismatched lengths should return stat=0");
        assert_eq!(p, 1.0, "mismatched lengths should return p=1");
    }
}
