use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grid::Grid;

fn init_vec_vec(size: usize) -> Vec<Vec<u32>> {
    vec![vec![0; size]; size]
}

fn init_vec_flat(size: usize) -> Vec<u32> {
    vec![0; size * size]
}

fn init_grid(size: usize) -> Grid<u32> {
    Grid::new(size, size, 0)
}

fn get_vec_vec(x: usize, y: usize) -> u32 {
    let mat = init_vec_vec(10);
    mat[x][y]
}

fn get_vec_flat(x: usize, y: usize) -> u32 {
    let mat = init_vec_flat(10);
    mat[x * 10 + y % 10]
}

fn get_grid(x: usize, y: usize) -> u32 {
    let mat = init_grid(10);
    mat[x][y]
}

fn get_grid_fn(x: usize, y: usize) {
    let mat = init_grid(10);
    mat.get(x, y);
}

fn set_vec_vec(x: usize, y: usize) {
    let mut mat = init_vec_vec(10);
    mat[x][y] = 10;
}
fn set_vec_flat(x: usize, y: usize) {
    let mut mat = init_vec_flat(10);
    mat[x * 10 + y % 10] = 10;
}

fn set_grid(x: usize, y: usize) {
    let mut mat = init_grid(10);
    mat[x][y] = 10;
}

fn criterion_benchmark(c: &mut Criterion) {
    // Init
    c.bench_function("Init vec vec 10x10", |b| {
        b.iter(|| init_vec_vec(black_box(10)))
    });
    c.bench_function("Init vec flat 10x10", |b| {
        b.iter(|| init_vec_flat(black_box(10)))
    });
    c.bench_function("Init grid 10x10", |b| b.iter(|| init_grid(black_box(10))));

    // Get
    c.bench_function("Get vec vec 10x10", |b| {
        b.iter(|| get_vec_vec(black_box(2), black_box(3)))
    });
    c.bench_function("Get vec flat 10x10", |b| {
        b.iter(|| get_vec_flat(black_box(2), black_box(3)))
    });
    c.bench_function("Get grid 10x10", |b| {
        b.iter(|| get_grid(black_box(2), black_box(3)))
    });
    c.bench_function("Get grid fn 10x10", |b| {
        b.iter(|| get_grid_fn(black_box(2), black_box(3)))
    });

    //Set
    c.bench_function("Set vec vec 10x10", |b| {
        b.iter(|| set_vec_vec(black_box(2), black_box(3)))
    });
    c.bench_function("Set vec flat 10x10", |b| {
        b.iter(|| set_vec_flat(black_box(2), black_box(3)))
    });
    c.bench_function("Set grid 10x10", |b| {
        b.iter(|| set_grid(black_box(2), black_box(3)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
