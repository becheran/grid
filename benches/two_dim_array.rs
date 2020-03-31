use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn init(x: usize, y: usize) -> Vec<Vec<u32>> {
    vec![vec![42; x]; y]
}

fn init_flat(x: usize, y: usize) -> Vec<u32> {
    vec![42; x * y]
}


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Init  20x20", |b| b.iter(|| init(black_box(20), black_box(20))));
    c.bench_function("Init flat  20x20", |b| b.iter(|| init_flat(black_box(20), black_box(20))));
    c.bench_function("Init  2000x2000", |b| b.iter(|| init(black_box(2000), black_box(2000))));
    c.bench_function("Init flat  2000x2000", |b| b.iter(|| init_flat(black_box(2000), black_box(2000))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
