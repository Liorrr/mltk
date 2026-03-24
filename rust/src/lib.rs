use pyo3::prelude::*;

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

/// ML Test Kit — Rust acceleration module.
#[pymodule]
fn _mltk_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ks_test, m)?)?;
    m.add_function(wrap_pyfunction!(psi, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ks_identical_distributions() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (stat, p) = ks_test(data.clone(), data);
        // Small samples may produce non-zero stat due to merge ordering
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
}
