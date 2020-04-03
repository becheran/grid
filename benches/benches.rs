use criterion::{black_box, criterion_group, criterion_main, Criterion};
use grid::Grid;

fn init_vec_vec(size: usize) -> Vec<Vec<u32>> {
    vec![vec![0; size]; size]
}

fn init_vec_flat(size: usize) -> Vec<u32> {
    vec![0; size * size]
}

fn init_grid(size: usize) -> Grid<u32> {
    Grid::init(size, size, 0)
}

fn criterion_benchmark(c: &mut Criterion) {
    // New
    c.bench_function("Init vec vec 10x10", |b| {
        b.iter(|| init_vec_vec(black_box(10)))
    });
    c.bench_function("Init vec flat 10x10", |b| {
        b.iter(|| init_vec_flat(black_box(10)))
    });
    c.bench_function("Init grid 10x10", |b| b.iter(|| init_grid(black_box(10))));

    // Get
    c.bench_function("Idx vec vec 10x10", |b| {
        let vec_vec = init_vec_vec(10);
        b.iter(|| vec_vec[black_box(2)][black_box(3)])
    });
    c.bench_function("Idx grid 10x10", |b| {
        let grid = init_grid(10);
        b.iter(|| grid[black_box(2)][black_box(3)])
    });
    c.bench_function("Get_fn vec vec 10x10", |b| {
        let vec_vec = init_vec_vec(10);
        b.iter(|| vec_vec.get(black_box(2)).unwrap().get(black_box(3)))
    });
    c.bench_function("Get_fn grid 10x10", |b| {
        let grid = init_grid(10);
        b.iter(|| grid.get(black_box(2), black_box(3)))
    });

    //Set
    c.bench_function("Set vec vec 10x10", |b| {
        let mut vec_vec = init_vec_vec(10);
        b.iter(|| vec_vec[black_box(2)][black_box(3)] = 2)
    });
    c.bench_function("Set gird 10x10", |b| {
        let mut gird = init_grid(10);
        b.iter(|| gird[black_box(2)][black_box(3)] = 2)
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
