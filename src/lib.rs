use std::ops::Index;
use std::ops::IndexMut;

pub struct Grid<T> {
    data: Vec<T>,
    cols: usize,
    rows: usize,
}

#[allow(unused_macros)]
macro_rules! count {
    () => (0usize);
    ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
}

#[macro_export]
macro_rules! grid {
    () => {
        Grid {
            rows: 0,
            cols: 0,
            data: vec![],
        }
    };
    ( [$( $x:expr ),* ]) => { {
        let vec = vec![$($x),*];
        Grid { rows : 1, cols: vec.len(), data: vec }
    } };
    ( [$( $x0:expr ),*] $([$( $x:expr ),*])* ) => {
        {
            let mut _assert_width0 = [(); count!($($x0)*)];
            let mut vec = Vec::new();
            let rows = 1usize;
            let cols = count!($($x0)*);

            $( vec.push($x0); )*

            $(
                let rows = rows + 1usize;
                let _assert_width = [(); count!($($x)*)];
                _assert_width0 = _assert_width;
                $( vec.push($x); )*
            )*

            Grid { rows : rows, cols: cols, data: vec }
        }
    };
}

impl<T: Clone> Grid<T> {
    /// Init a grid of size rows x columns with default values of the given type.
    /// For example this will generate a 2x3 grid of zeros:
    /// ```
    /// use grid::Grid;
    /// let grid : Grid<u8> = Grid::new(2,2);
    /// assert_eq!(grid[0][0], 0);
    /// ```
    pub fn new(rows: usize, cols: usize) -> Grid<T>
    where
        T: Default,
    {
        if rows < 1 || cols < 1 {
            panic!("Grid size of rows and columns must be greater than zero.");
        }
        Grid {
            data: vec![T::default(); rows * cols],
            cols: cols,
            rows: rows,
        }
    }

    /// Init a grid of size rows x columns with the given data element.
    pub fn init(rows: usize, cols: usize, data: T) -> Grid<T> {
        if rows < 1 || cols < 1 {
            panic!("Grid size of rows and columns must be greater than zero.");
        }
        Grid {
            data: vec![data; rows * cols],
            cols: cols,
            rows: rows,
        }
    }

    /// Access a certain element in the grid.
    /// Returns None if an element beyond the grid bounds is tried to be accessed.
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.data.get(row * self.cols + column % self.cols)
    }

    /// Returns the size of the gird as a two element tuple.
    /// First element are the number of rows and the second the columns.
    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.cols)
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
        &self.data[(idx * &self.cols)..]
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
        &mut self.data[(idx * &self.cols)..]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn macro_init() {
        let grid = grid![[1, 2, 3][4, 5, 6]];
        assert_eq!(grid[0][0], 1);
        assert_eq!(grid[0][1], 2);
        assert_eq!(grid[0][2], 3);
        assert_eq!(grid[1][0], 4);
        assert_eq!(grid[1][1], 5);
        assert_eq!(grid[1][2], 6);
    }

    #[test]
    fn macro_init_2() {
        let grid = grid![[1, 2, 3]
                         [4, 5, 6]
                         [7,8,9]];
        assert_eq!(grid.size(), (3, 3))
    }

    #[test]
    fn macro_one_row() {
        let grid: Grid<usize> = grid![[1, 2, 3, 4]];
        assert_eq!(grid.size(), (1, 4));
        assert_eq!(grid[0][0], 1);
        assert_eq!(grid[0][1], 2);
        assert_eq!(grid[0][2], 3);
    }

    #[test]
    fn macro_init_empty() {
        let grid: Grid<usize> = grid![];
        assert_eq!(grid.size(), (0, 0));
    }

    #[test]
    fn init() {
        Grid::init(1, 2, 3);
        Grid::init(1, 2, 1.2);
        Grid::init(1, 2, 'a');
    }

    #[test]
    fn new() {
        let grid: Grid<u8> = Grid::new(1, 2);
        assert_eq!(grid[0][0], 0);
    }

    #[test]
    #[should_panic]
    fn init_panics() {
        Grid::init(0, 2, 3);
    }

    #[test]
    #[should_panic]
    fn ctr_panics_2() {
        Grid::init(1, 0, 3);
    }

    #[test]
    fn get() {
        let grid = Grid::init(1, 2, 3);
        assert_eq!(grid.get(0, 0), Some(&3));
    }
    #[test]
    fn get_none() {
        let grid = Grid::init(1, 2, 3);
        assert_eq!(grid.get(1, 0), None);
    }

    #[test]
    fn idx() {
        let grid = Grid::init(1, 2, 3);
        assert_eq!(grid[0][0], 3);
    }

    #[test]
    #[should_panic]
    fn idx_panic_1() {
        let grid = Grid::init(1, 2, 3);
        grid[20][0];
    }

    #[test]
    #[should_panic]
    fn idx_panic_2() {
        let grid = Grid::init(1, 2, 3);
        grid[0][20];
    }

    #[test]
    fn idx_set() {
        let mut grid = Grid::init(1, 2, 3);
        grid[0][0] = 4;
        assert_eq!(grid[0][0], 4);
    }

    #[test]
    fn size() {
        let grid = Grid::init(1, 2, 3);
        assert_eq!(grid.size(), (1, 2));
    }
}
