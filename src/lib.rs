use std::ops::Index;
use std::ops::IndexMut;

/// Stores elements of a certain type in a 2D grid structure.
pub struct Grid<T> {
    data: Vec<T>,
    cols: usize,
    rows: usize,
}

#[doc(hidden)]
#[macro_export]
macro_rules! count {
    () => (0usize);
    ( $x:tt $($xs:tt)* ) => (1usize + $crate::count!($($xs)*));
}

/// Init a grid with values.
/// ```
/// use grid::grid;
/// let grid = grid![[1, 2, 3]
/// [4, 5, 6]
/// [7, 8, 9]];
/// assert_eq!(grid.size(), (3, 3))
/// ```
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
        $crate::Grid::from_vec(&vec, vec.len())
    } };
    ( [$( $x0:expr ),*] $([$( $x:expr ),*])* ) => {
        {
            let mut _assert_width0 = [(); $crate::count!($($x0)*)];
            let mut vec = Vec::new();
            let cols = $crate::count!($($x0)*);

            $( vec.push($x0); )*

            $(
                let _assert_width = [(); $crate::count!($($x)*)];
                _assert_width0 = _assert_width;
                $( vec.push($x); )*
            )*

            $crate::Grid::from_vec(&vec, cols)
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

    /// Returns a grid from a vector with a given column length.
    /// The length of `vec` must be a multiple of `cols`.
    ///
    /// For example:
    ///
    /// ```
    /// use grid::Grid;
    /// let grid = Grid::from_vec(&vec![1,2,3,4,5,6], 3);
    /// assert_eq!(grid.size(), (2, 3));
    /// ```
    ///
    /// will create a grid with the following layout:
    /// [1,2,3]
    /// [4,5,6]
    ///
    /// This example will fail, because `vec.len()` is not a multiple of `cols`:
    ///
    /// ``` should_panic
    /// use grid::Grid;
    /// Grid::from_vec(&vec![1,2,3,4,5], 3);
    /// ```
    pub fn from_vec(vec: &Vec<T>, cols: usize) -> Grid<T> {
        if vec.len() == 0 {
            if cols == 0 {
                return grid![];
            } else {
                panic!("Vector length is zero, but cols is {:?}", cols);
            }
        }
        if vec.len() % cols != 0 {
            panic!("Vector length must be a multiple of cols.");
        }
        Grid {
            data: vec.to_vec(),
            rows: vec.len() / cols,
            cols: cols,
        }
    }

    /// Access a certain element in the grid.
    /// Returns None if an element beyond the grid bounds is tried to be accessed.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols() {
            unsafe { Some(&self.data.get_unchecked(row * self.cols() + col)) }
            //Some(&self.data.get_unchecked(row * self.cols() + col))
        } else {
            None
        }
    }

    /// Returns the size of the gird as a two element tuple.
    /// First element are the number of rows and the second the columns.
    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the number of rows of the grid.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns of the grid.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl<T: Clone> Index<usize> for Grid<T> {
    type Output = [T];

    fn index(&self, idx: usize) -> &Self::Output {
        if idx < self.rows {
            &self.data[(idx * &self.cols)..]
        } else {
            panic!(
                "index {:?} out of bounds. Grid has {:?} rows.",
                self.rows, idx
            );
        }
    }
}

impl<T: Clone> IndexMut<usize> for Grid<T> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
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
                         [7, 8, 9]];
        assert_eq!(grid.size(), (3, 3))
    }

    #[test]
    fn macro_init_char() {
        let grid = grid![['a', 'b', 'c']
                         ['a', 'b', 'c']
                         ['a', 'b', 'c']];
        assert_eq!(grid.size(), (3, 3));
        assert_eq!(grid[1][1], 'b');
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
    fn from_vec_zero() {
        let grid: Grid<u8> = Grid::from_vec(&vec![], 0);
        assert_eq!(grid.size(), (0, 0));
    }

    #[test]
    #[should_panic]
    fn from_vec_panics_1() {
        let _: Grid<u8> = Grid::from_vec(&vec![1, 2, 3], 0);
    }

    #[test]
    #[should_panic]
    fn from_vec_panics_2() {
        let _: Grid<u8> = Grid::from_vec(&vec![1, 2, 3], 2);
    }

    #[test]
    #[should_panic]
    fn from_vec_panics_3() {
        let _: Grid<u8> = Grid::from_vec(&vec![], 1);
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
