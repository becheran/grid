use criterion::{criterion_group, criterion_main, Criterion};
use grid::grid;
use grid::Grid;
use rand::Rng;

const SIZE: usize = 1000;

fn init_vec_vec() -> Vec<Vec<u32>> {
    vec![vec![0; SIZE]; SIZE]
}

fn init_vec_flat() -> Vec<u32> {
    vec![0; SIZE * SIZE]
}

fn init_grid() -> Grid<u32> {
    Grid::init(SIZE, SIZE, 0)
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut rand = || rng.gen_range(0, SIZE);

    let mut rng_u32 = rand::thread_rng();
    let mut rand_u32 = || rng_u32.gen::<u32>();

    // Init macro
    c.bench_function("Macro init vec vec", |b| {
        b.iter(|| {
            vec![
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ]
        })
    });
    c.bench_function("Macro init vec flat", |b| {
        b.iter(|| {
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3,
                4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            ]
        })
    });
    c.bench_function("Macro init grid from_vec", |b| {
        b.iter(|| {
            let vec = vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3,
                4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            ];
            Grid::from_vec(vec, 10)
        })
    });
    c.bench_function("Macro init grid", |b| {
        b.iter(|| {
            grid![[0,1,2,3,4,5,6,7,8,9]
            [0,1,2,3,4,5,6,7,8,9]
            [0,1,2,3,4,5,6,7,8,9]
            [0,1,2,3,4,5,6,7,8,9]
            [0,1,2,3,4,5,6,7,8,9]
            [0,1,2,3,4,5,6,7,8,9]
            [0,1,2,3,4,5,6,7,8,9]
            [0,1,2,3,4,5,6,7,8,9]
            [0,1,2,3,4,5,6,7,8,9]
            [0,1,2,3,4,5,6,7,8,9]]
        })
    });

    // New
    c.bench_function("Init vec vec", |b| b.iter(|| init_vec_vec()));
    c.bench_function("Init vec flat", |b| b.iter(|| init_vec_flat()));
    c.bench_function("Init grid", |b| b.iter(|| init_grid()));

    // Get
    c.bench_function("Idx vec vec", |b| {
        let vec_vec = init_vec_vec();
        b.iter(|| vec_vec[rand()][rand()])
    });
    c.bench_function("Idx grid", |b| {
        let grid = init_grid();
        b.iter(|| grid[rand()][rand()])
    });
    c.bench_function("Get_fn vec vec", |b| {
        let vec_vec = init_vec_vec();
        b.iter(|| vec_vec.get(rand()).unwrap().get(rand()))
    });
    c.bench_function("Get_fn grid", |b| {
        let grid = init_grid();
        b.iter(|| grid.get(rand(), rand()))
    });

    //Set
    c.bench_function("Set vec vec", |b| {
        let mut vec_vec = init_vec_vec();
        b.iter(|| vec_vec[rand()][rand()] = rand_u32())
    });
    c.bench_function("Set gird", |b| {
        let mut gird = init_grid();
        b.iter(|| gird[rand()][rand()] = rand_u32())
    });

    // Push
    c.bench_function("Push row grid", |b| {
        let mut grid = init_grid();
        b.iter(|| grid.push_row(vec![10; SIZE]))
    });
    c.bench_function("Push col grid", |b| {
        let mut gird = init_grid();
        b.iter(|| gird.push_col(vec![10; SIZE]))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
