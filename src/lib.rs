use std::ops::Index;

pub struct Grid<T> {
    data: Vec<T>,
    row_len: usize,
}

impl<T: Clone> Grid<T> {
    pub fn new(rows: usize, columns: usize, data: T) -> Grid<T> {
        Grid {
            data: vec![data; rows * columns],
            row_len: rows,
        }
    }

    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.data.get(row / self.row_len + column % self.row_len)
    }
}

impl<T: Clone> Index<usize> for Grid<T> {
    type Output = [T];

    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[(idx / &self.row_len)..]
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
    fn get() {
        let grid = Grid::new(1, 2, 3);
        assert_eq!(grid.get(0, 0), Some(&3));
    }

    #[test]
    fn idx() {
        let grid = Grid::new(1, 2, 3);
        assert_eq!(grid[0][0], 3);
    }
}
