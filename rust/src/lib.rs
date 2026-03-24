use pyo3::prelude::*;

/// Compute KS test statistic between two distributions.
/// Returns (statistic, p_value).
#[pyfunction]
fn ks_test(reference: Vec<f64>, current: Vec<f64>) -> (f64, f64) {
    // TODO: Sprint 2 — implement KS test
    let _ = (&reference, &current);
    (0.0, 1.0)
}

/// Compute Population Stability Index between two distributions.
#[pyfunction]
fn psi(reference: Vec<f64>, current: Vec<f64>, bins: usize) -> f64 {
    // TODO: Sprint 2 — implement PSI
    let _ = (&reference, &current, bins);
    0.0
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
    fn test_ks_test_placeholder() {
        let (stat, p) = ks_test(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        assert_eq!(stat, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_psi_placeholder() {
        let result = psi(vec![1.0, 2.0], vec![1.0, 2.0], 10);
        assert_eq!(result, 0.0);
    }
}
