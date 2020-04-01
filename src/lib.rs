use std::ops::Index;
use std::ops::IndexMut;

pub struct Grid<T> {
    data: Vec<T>,
    columns: usize,
    rows: usize,
}

impl<T: Clone> Grid<T> {
    pub fn new(rows: usize, columns: usize, data: T) -> Grid<T> {
        if rows < 1 || columns < 1 {
            panic!("Grid size of rows and columns must be greater than zero.");
        }
        Grid {
            data: vec![data; rows * columns],
            columns: columns,
            rows: rows,
        }
    }

    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.data.get(row * self.columns + column % self.columns)
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
}

impl<T: Clone> Index<usize> for Grid<T> {
    type Output = [T];

    fn index(&self, idx: usize) -> &Self::Output {
        if idx >= self.rows {
            panic!(
                "index out of bounds: grid has {:?} but the index is {:?}",
                self.rows, idx
            );
        }
        &self.data[(idx * &self.columns)..]
    }
}

impl<T: Clone> IndexMut<usize> for Grid<T> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        if idx >= self.rows {
            panic!(
                "index out of bounds: grid has {:?} but the index is {:?}",
                self.rows, idx
            );
        }
        &mut self.data[(idx * &self.columns)..]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ctr() {
        Grid::new(1, 2, 3);
        Grid::new(1, 2, 1.2);
        Grid::new(1, 2, 'a');
    }
    #[test]
    #[should_panic]
    fn ctr_panics() {
        Grid::new(0, 2, 3);
    }

    #[test]
    #[should_panic]
    fn ctr_panics_2() {
        Grid::new(1, 0, 3);
    }

    #[test]
    fn get() {
        let grid = Grid::new(1, 2, 3);
        assert_eq!(grid.get(0, 0), Some(&3));
    }
    #[test]
    fn get_none() {
        let grid = Grid::new(1, 2, 3);
        assert_eq!(grid.get(1, 0), None);
    }

    #[test]
    fn idx() {
        let grid = Grid::new(1, 2, 3);
        assert_eq!(grid[0][0], 3);
    }

    #[test]
    #[should_panic]
    fn idx_panic_1() {
        let grid = Grid::new(1, 2, 3);
        grid[20][0];
    }

    #[test]
    #[should_panic]
    fn idx_panic_2() {
        let grid = Grid::new(1, 2, 3);
        grid[0][20];
    }

    #[test]
    fn idx_set() {
        let mut grid = Grid::new(1, 2, 3);
        grid[0][0] = 4;
        assert_eq!(grid[0][0], 4);
    }

    #[test]
    fn size() {
        let grid = Grid::new(1, 2, 3);
        assert_eq!(grid.size(), (1, 2));
    }
}
