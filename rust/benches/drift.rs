use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_ks_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("ks_test");
    for size in [1_000, 10_000, 100_000] {
        let ref_data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
        let cur_data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.001) + 0.5).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                _mltk_rust::ks_test_core(
                    black_box(&mut ref_data.clone()),
                    black_box(&mut cur_data.clone()),
                )
            });
        });
    }
    group.finish();
}

fn bench_psi(c: &mut Criterion) {
    let mut group = c.benchmark_group("psi");
    for size in [1_000, 10_000, 100_000] {
        let ref_data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
        let cur_data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.001) + 0.5).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| _mltk_rust::psi_core(black_box(&ref_data), black_box(&cur_data), 10));
        });
    }
    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    for size in [1_000, 10_000, 100_000] {
        let a: Vec<f64> = (0..size).map(|i| (i as f64 * 0.001).sin()).collect();
        let b: Vec<f64> = (0..size).map(|i| (i as f64 * 0.001).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b_iter, _| {
            b_iter.iter(|| {
                _mltk_rust::cosine_similarity_core(black_box(&a), black_box(&b))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ks_test, bench_psi, bench_cosine_similarity);
criterion_main!(benches);
