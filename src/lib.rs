#![warn(clippy::all, clippy::pedantic)]

/*!
# Two Dimensional Grid
Continuous growable 2D data structure.
The purpose of this crate is to provide an universal data structure that is faster,
uses less memory, and is easier to use than a naive `Vec<Vec<T>>` solution.

Similar to *C-like* arrays `grid` uses a flat 1D `Vec<T>` data structure to have a continuous
memory data layout. See also [this](https://stackoverflow.com/questions/17259877/1d-or-2d-array-whats-faster)
explanation of why you should probably use a one-dimensional array approach.

Note that this crate uses a [*row-major*](https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays) memory layout.
Therefore, `grid.push_row()` is way faster then the `grid.push_col()` operation.

This crate will always provide a 2D data structure. If you need three or more dimensions take a look at the
[ndarray](https://docs.rs/ndarray/0.13.0/ndarray/) library. The `grid` create is a container for all kind of data.
If you need to perform matrix operations, you are better of with a linear algebra lib, such as
[cgmath](https://docs.rs/cgmath/0.17.0/cgmath/) or [nalgebra](https://docs.rs/nalgebra/0.21.0/nalgebra/).
No other dependencies except for the std lib are used.
Most of the functions `std::Vec<T>` offer are also implemented in `grid` and slightly modified for a 2D data object.
# Examples
```
use grid::*;
let mut grid = grid![[1,2,3]
                     [4,5,6]];
assert_eq!(grid, Grid::from_vec(vec![1,2,3,4,5,6],3));
assert_eq!(grid.get(0,2), Some(&3));
assert_eq!(grid[1][1], 5);
assert_eq!(grid.size(), (2,3));
grid.push_row(vec![7,8,9]);
assert_eq!(grid, grid![[1,2,3][4,5,6][7,8,9]])
 ```
*/

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(not(feature = "std")))]
extern crate alloc;
#[cfg(all(not(feature = "std")))]
use alloc::{format, vec, vec::Vec};

use core::cmp;
use core::cmp::Eq;
use core::fmt;
use core::iter::StepBy;
use core::ops::Index;
use core::ops::IndexMut;
use core::slice::Iter;
use core::slice::IterMut;

#[doc(hidden)]
#[macro_export]
macro_rules! count {
    () => (0usize);
    ( $x:tt $($xs:tt)* ) => (1usize + $crate::count!($($xs)*));
}

/// Init a grid with values.
///
/// Each array within `[]` represents a row starting from top to button.
///
/// # Examples
///
/// In this example a grid of numbers from 1 to 9 is created:
///
///  
/// ```
/// use grid::grid;
/// let grid = grid![[1, 2, 3]
/// [4, 5, 6]
/// [7, 8, 9]];
/// assert_eq!(grid.size(), (3, 3))
/// ```
///
/// # Examples
///
/// Not that each row must be of the same length. The following example will not compile:  
///  
/// ``` ignore
/// use grid::grid;
/// let grid = grid![[1, 2, 3]
/// [4, 5] // This does not work!
/// [7, 8, 9]];
/// ```
#[macro_export]
macro_rules! grid {
    () => {
        $crate::Grid::from_vec(vec![], 0)
    };
    ( [$( $x:expr ),* ]) => { {
        let vec = vec![$($x),*];
        let len  = vec.len();
        $crate::Grid::from_vec(vec, len)
    } };
    ( [$( $x0:expr ),*] $([$( $x:expr ),*])* ) => {
        {
            let mut _assert_width0 = [(); $crate::count!($($x0)*)];
            let cols = $crate::count!($($x0)*);
            let rows = 1usize;

            $(
                let _assert_width = [(); $crate::count!($($x)*)];
                _assert_width0 = _assert_width;
                let rows = rows + 1usize;
            )*

            let mut vec = Vec::with_capacity(rows.checked_mul(cols).unwrap());

            $( vec.push($x0); )*
            $( $( vec.push($x); )* )*

            $crate::Grid::from_vec(vec, cols)
        }
    };
}

/// Stores elements of a certain type in a 2D grid structure.
///
/// Uses a rust `Vec<T>` type to reference the grid data on the heap.
/// Also the number of rows and columns are stored in the grid data structure.
///
/// The size limit of a grid is `rows * cols < usize`.
///
/// The grid data is stored in a row-major memory layout.
pub struct Grid<T> {
    data: Vec<T>,
    cols: usize,
    rows: usize,
}

impl<T> Grid<T> {
    /// Init a grid of size rows x columns with default values of the given type.
    /// For example this will generate a 2x3 grid of zeros:
    ///
    /// ```
    /// use grid::Grid;
    /// let grid : Grid<u8> = Grid::new(2,3);
    /// assert_eq!(grid[0][0], 0);
    /// ```
    ///
    /// If `rows == 0` or `cols == 0` the grid will be empty with no cols and rows.
    ///
    /// # Panics
    ///
    /// Panics if `rows * cols > usize`.
    pub fn new(rows: usize, cols: usize) -> Grid<T>
    where
        T: Default,
    {
        if rows == 0 || cols == 0 {
            return Grid {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            };
        }
        let mut data = Vec::new();
        data.resize_with(rows.checked_mul(cols).unwrap(), T::default);
        Grid { data, cols, rows }
    }

    /// Init a grid of size rows x columns with the given data element.
    ///
    /// If `rows == 0` or `cols == 0` the grid will be empty with no cols and rows.
    ///
    /// # Panics
    ///
    /// Panics if `rows * cols > usize`.
    pub fn init(rows: usize, cols: usize, data: T) -> Grid<T>
    where
        T: Clone,
    {
        if rows == 0 || cols == 0 {
            return Grid {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            };
        }
        Grid {
            data: vec![data; rows.checked_mul(cols).unwrap()],
            cols,
            rows,
        }
    }

    /// Returns a grid from a vector with a given column length.
    /// The length of `vec` must be a multiple of `cols`.
    ///
    /// For example:
    ///
    /// ```
    /// use grid::Grid;
    /// let grid = Grid::from_vec(vec![1,2,3,4,5,6], 3);
    /// assert_eq!(grid.size(), (2, 3));
    /// ```
    ///
    /// will create a grid with the following layout:
    /// \[1,2,3\]
    /// \[4,5,6\]
    ///
    /// This example will fail, because `vec.len()` is not a multiple of `cols`:
    ///
    /// ``` should_panic
    /// use grid::Grid;
    /// Grid::from_vec(vec![1,2,3,4,5], 3);
    /// ```
    ///
    /// # Panics
    ///
    /// This panics if the vector length isn't a multiple of the number of columns.
    #[must_use]
    pub fn from_vec(vec: Vec<T>, cols: usize) -> Grid<T> {
        let rows = vec.len().checked_div(cols).unwrap_or(0);
        assert_eq!(
            rows * cols,
            vec.len(),
            "Vector length {:?} should be a multiple of cols = {:?}",
            vec.len(),
            cols
        );
        if rows == 0 || cols == 0 {
            Grid {
                data: vec,
                rows: 0,
                cols: 0,
            }
        } else {
            Grid {
                data: vec,
                rows,
                cols,
            }
        }
    }

    /// Returns a reference to an element, without performing bound checks.
    /// Generally not recommended, use with caution!
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior even if the resulting reference is not used.
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> &T {
        self.data.get_unchecked(row * self.cols + col)
    }

    /// Returns a mutable reference to an element, without performing bound checks.
    /// Generally not recommended, use with caution!
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior even if the resulting reference is not used.
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut T {
        self.data.get_unchecked_mut(row * self.cols + col)
    }

    /// Access a certain element in the grid.
    /// Returns None if an element beyond the grid bounds is tried to be accessed.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            unsafe { Some(self.get_unchecked(row, col)) }
        } else {
            None
        }
    }

    /// Mutable access to a certain element in the grid.
    /// Returns None if an element beyond the grid bounds is tried to be accessed.
    #[must_use]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            unsafe { Some(self.get_unchecked_mut(row, col)) }
        } else {
            None
        }
    }

    /// Returns the size of the gird as a two element tuple.
    /// First element are the number of rows and the second the columns.
    #[must_use]
    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the number of rows of the grid.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns of the grid.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns true if the grid contains no elements.
    /// For example:
    /// ```
    /// use grid::*;
    /// let grid : Grid<u8> = grid![];
    /// assert!(grid.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clears the grid.
    pub fn clear(&mut self) {
        self.rows = 0;
        self.cols = 0;
        self.data.clear();
    }

    /// Returns an iterator over the whole grid, starting from the first row and column.
    /// ```
    /// use grid::*;
    /// let grid: Grid<u8> = grid![[1,2][3,4]];
    /// let mut iter = grid.iter();
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), Some(&4));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<T> {
        self.data.iter()
    }

    /// Returns an mutable iterator over the whole grid that allows modifying each value.
    /// ```
    /// use grid::*;
    /// let mut grid: Grid<u8> = grid![[1,2][3,4]];
    /// let mut iter = grid.iter_mut();
    /// let next = iter.next();
    /// assert_eq!(next, Some(&mut 1));
    /// *next.unwrap() = 10;
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<T> {
        self.data.iter_mut()
    }

    /// Returns an iterator over a column.
    ///
    /// # Examples
    ///
    /// ```
    /// use grid::*;
    /// let grid: Grid<u8> = grid![[1, 2, 3][3, 4, 5]];
    /// let mut col_iter = grid.iter_col(1);
    /// assert_eq!(col_iter.next(), Some(&2));
    /// assert_eq!(col_iter.next(), Some(&4));
    /// assert_eq!(col_iter.next(), None);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the col index is out of bounds.
    pub fn iter_col(&self, col: usize) -> StepBy<Iter<T>> {
        if col < self.cols {
            return self.data[col..].iter().step_by(self.cols);
        }
        panic!(
            "out of bounds. Column must be less than {:?}, but is {:?}.",
            self.cols, col
        )
    }

    /// Returns a mutable iterator over a column.
    ///
    /// # Examples
    ///
    /// ```
    /// use grid::*;
    /// let mut grid: Grid<u8> = grid![[1, 2, 3][3, 4, 5]];
    /// let mut col_iter = grid.iter_col_mut(1);
    /// let next = col_iter.next();
    /// assert_eq!(next, Some(&mut 2));
    /// *next.unwrap() = 10;
    /// assert_eq!(grid[0][1], 10);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the col index is out of bounds.
    pub fn iter_col_mut(&mut self, col: usize) -> StepBy<IterMut<T>> {
        let cols = self.cols;
        if col < cols {
            return self.data[col..].iter_mut().step_by(cols);
        }
        panic!(
            "out of bounds. Column must be less than {:?}, but is {:?}.",
            self.cols, col
        )
    }

    /// Returns an iterator over a row.
    ///
    /// # Examples
    ///
    /// ```
    /// use grid::*;
    /// let grid: Grid<u8> = grid![[1, 2, 3][3, 4, 5]];
    /// let mut col_iter = grid.iter_row(1);
    /// assert_eq!(col_iter.next(), Some(&3));
    /// assert_eq!(col_iter.next(), Some(&4));
    /// assert_eq!(col_iter.next(), Some(&5));
    /// assert_eq!(col_iter.next(), None);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the row index is out of bounds.
    pub fn iter_row(&self, row: usize) -> Iter<T> {
        if row < self.rows {
            let start = row * self.cols;
            self.data[start..(start + self.cols)].iter()
        } else {
            panic!(
                "out of bounds. Row must be less than {:?}, but is {:?}.",
                self.rows, row
            )
        }
    }

    /// Returns a mutable iterator over a row.
    ///
    /// # Examples
    ///
    /// ```
    /// use grid::*;
    /// let mut grid: Grid<u8> = grid![[1, 2, 3][3, 4, 5]];
    /// let mut col_iter = grid.iter_row_mut(1);
    /// let next = col_iter.next();
    /// *next.unwrap() = 10;
    /// assert_eq!(grid[1][0], 10);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the row index is out of bounds.
    pub fn iter_row_mut(&mut self, row: usize) -> IterMut<T> {
        if row < self.rows {
            let cols = self.cols;
            let start = row * cols;
            self.data[start..(start + cols)].iter_mut()
        } else {
            panic!(
                "out of bounds. Row must be less than {:?}, but is {:?}.",
                self.rows, row
            )
        }
    }

    /// Traverse the grid with row and column indexes.
    /// 
    /// 
    /// # Examples
    /// 
    /// ```
    /// use grid::*;
    /// let grid: Grid<u8> = grid![[1,2][3,4]];
    /// let mut iter = grid.indexed_iter();
    /// assert_eq!(iter.next(), Some(((0, 0), &1)));
    /// 
    /// ```
    /// 
    /// Or simply unpack in a `for`  loop
    /// 
    /// ```
    /// use grid::*;
    /// let grid: Grid<u8> = grid![[1,2][3,4]];
    /// for ((row, col), i) in grid.indexed_iter() {
    ///     println!("value at row {row} and column {col} is: {i}");
    /// }
    /// ```
    /// 
    pub fn indexed_iter(&self) -> GridIndexedIter<'_, T> {
        GridIndexedIter { grid: &self, index: 0 }
    }

    /// Add a new row to the grid.
    ///
    /// # Examples
    ///
    /// ```
    /// use grid::*;
    /// let mut grid: Grid<u8> = grid![[1, 2, 3][3, 4, 5]];
    /// let row = vec![6,7,8];
    /// grid.push_row(row);
    /// assert_eq!(grid.rows(), 3);
    /// assert_eq!(grid[2][0], 6);
    /// assert_eq!(grid[2][1], 7);
    /// assert_eq!(grid[2][2], 8);
    /// ```
    ///
    /// Can also be used to init an empty grid:
    ///
    /// ```
    /// use grid::*;
    /// let mut grid: Grid<u8> = grid![];
    /// let row = vec![1,2,3];
    /// grid.push_row(row);
    /// assert_eq!(grid.size(), (1, 3));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    ///  - the grid is not empty and `row.len() != grid.cols()`
    ///  - `row.len() == 0`
    pub fn push_row(&mut self, row: Vec<T>) {
        assert_ne!(row.len(), 0);
        assert!(
            !(self.rows > 0 && row.len() != self.cols),
            "pushed row does not match. Length must be {:?}, but was {:?}.",
            self.cols,
            row.len()
        );
        self.data.extend(row);
        self.rows += 1;
        if self.cols == 0 {
            self.cols = self.data.len();
        }
    }

    /// Add a new column to the grid.
    ///
    /// *Important:*
    /// Please note that `Grid` uses a Row-Major memory layout. Therefore, the `push_col()`
    /// operation will be significantly slower compared to a `push_row()` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use grid::*;
    /// let mut grid: Grid<u8> = grid![[1, 2, 3][3, 4, 5]];
    /// let col = vec![4,6];
    /// grid.push_col(col);
    /// assert_eq!(grid.cols(), 4);
    /// assert_eq!(grid[0][3], 4);
    /// assert_eq!(grid[1][3], 6);
    /// ```
    ///
    /// Can also be used to init an empty grid:
    ///
    /// ```
    /// use grid::*;
    /// let mut grid: Grid<u8> = grid![];
    /// let col = vec![1,2,3];
    /// grid.push_col(col);
    /// assert_eq!(grid.size(), (3, 1));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    ///  - the grid is not empty and `col.len() != grid.rows()`
    ///  - `col.len() == 0`
    pub fn push_col(&mut self, col: Vec<T>) {
        assert_ne!(col.len(), 0);
        assert!(
            !(self.cols > 0 && col.len() != self.rows),
            "pushed column does not match. Length must be {:?}, but was {:?}.",
            self.rows,
            col.len()
        );
        self.data.extend(col);
        for i in (1..self.rows).rev() {
            let row_idx = i * self.cols;
            self.data[row_idx..row_idx + self.cols + i].rotate_right(i);
        }
        self.cols += 1;
        if self.rows == 0 {
            self.rows = self.data.len();
        }
    }

    /// Removes the last row from a grid and returns it, or None if it is empty.
    ///
    /// # Examples
    /// ```
    /// use grid::*;
    /// let mut grid = grid![[1,2,3][4,5,6]];
    /// assert_eq![grid.pop_row(), Some(vec![4,5,6])];
    /// assert_eq![grid.pop_row(), Some(vec![1,2,3])];
    /// assert_eq![grid.pop_row(), None];
    /// ```
    pub fn pop_row(&mut self) -> Option<Vec<T>> {
        if self.rows == 0 {
            return None;
        }
        let row = self.data.split_off((self.rows - 1) * self.cols);
        self.rows -= 1;
        if self.rows == 0 {
            self.cols = 0;
        }
        Some(row)
    }

    /// Remove a Row at the index and return a vector of it.
    ///
    /// # Examples
    /// ```
    /// use grid::*;
    /// let mut grid = grid![[1,2][3,4][5,6]];
    /// assert_eq![grid.remove_row(1), Some(vec![3,4])];   
    /// assert_eq![grid.remove_row(0), Some(vec![1,2])];
    /// assert_eq![grid.remove_row(0), Some(vec![5,6])];
    /// assert_eq![grid.remove_row(0), None];
    /// ```
    pub fn remove_row(&mut self, row_index: usize) -> Option<Vec<T>> {
        if self.cols == 0 || self.rows == 0 || row_index >= self.rows {
            return None;
        }
        let residue = self
            .data
            .drain((row_index * self.cols)..((row_index + 1) * self.cols));

        self.rows -= 1;
        if self.rows == 0 {
            self.cols = 0;
        }
        Some(residue.collect())
    }

    /// Removes the last column from a grid and returns it, or None if it is empty.
    ///
    /// Note that this operation is much slower than the `pop_row()` because the memory layout
    /// of `Grid` is row-major and removing a column requires a lot of move operations.
    ///
    /// # Examples
    /// ```
    /// use grid::*;
    /// let mut grid = grid![[1,2,3][4,5,6]];
    /// assert_eq![grid.pop_col(), Some(vec![3,6])];
    /// assert_eq![grid.pop_col(), Some(vec![2,5])];
    /// assert_eq![grid.pop_col(), Some(vec![1,4])];
    /// assert_eq![grid.pop_col(), None];
    /// ```
    pub fn pop_col(&mut self) -> Option<Vec<T>> {
        if self.cols == 0 {
            return None;
        }
        for i in 1..self.rows {
            let row_idx = i * (self.cols - 1);
            self.data[row_idx..row_idx + self.cols + i - 1].rotate_left(i);
        }
        let col = self.data.split_off(self.data.len() - self.rows);
        self.cols -= 1;
        if self.cols == 0 {
            self.rows = 0;
        }
        Some(col)
    }

    /// Remove a column at the index and return a vector of it.
    ///
    /// # Examples
    /// ```
    /// use grid::*;
    /// let mut grid = grid![[1,2,3,4][5,6,7,8][9,10,11,12][13,14,15,16]];
    /// assert_eq![grid.remove_col(3), Some(vec![4,8,12,16])];
    /// assert_eq![grid.remove_col(0), Some(vec![1,5,9,13])];
    /// assert_eq![grid.remove_col(1), Some(vec![3,7,11,15])];
    /// assert_eq![grid.remove_col(0), Some(vec![2,6,10,14])];
    /// assert_eq![grid.remove_col(0), None];
    /// ```
    pub fn remove_col(&mut self, col_index: usize) -> Option<Vec<T>> {
        if self.cols == 0 || self.rows == 0 || col_index >= self.cols {
            return None;
        }
        for i in 0..self.rows {
            let row_idx = col_index + i * (self.cols - 1);
            let end = cmp::min(row_idx + self.cols + i, self.data.len());
            self.data[row_idx..end].rotate_left(i + 1);
        }
        let col = self.data.split_off(self.data.len() - self.rows);
        self.cols -= 1;
        if self.cols == 0 {
            self.rows = 0;
        }
        Some(col)
    }

    /// Insert a new row at the index and shifts all rows after down.
    ///
    /// # Examples
    /// ```
    /// use grid::*;
    /// let mut grid = grid![[1,2,3][4,5,6]];
    /// grid.insert_row(1, vec![7,8,9]);
    /// assert_eq!(grid[0], [1,2,3]);
    /// assert_eq!(grid[1], [7,8,9]);
    /// assert_eq!(grid[2], [4,5,6]);
    /// assert_eq!(grid.size(), (3,3))
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - the grid is not empty and `row.len() != grid.cols()`.
    /// - the index is greater than the number of rows
    pub fn insert_row(&mut self, index: usize, row: Vec<T>) {
        let input_len = row.len();
        assert!(
            !(self.cols > 0 && input_len != self.cols),
            "Inserted row must be of length {}, but was {}.",
            self.cols,
            row.len()
        );
        assert!(
            index <= self.rows,
            "Out of range. Index was {}, but must be less or equal to {}.",
            index,
            self.rows
        );
        let data_idx = index * input_len;
        self.data.splice(data_idx..data_idx, row.into_iter());
        self.cols = input_len;
        self.rows += 1;
    }

    /// Insert a new column at the index.
    ///
    /// Important! Insertion of columns is a lot slower than the lines insertion.
    /// This is because of the memory layout of the grid data structure.
    ///
    /// # Examples
    /// ```
    /// use grid::*;
    /// let mut grid = grid![[1,2,3][4,5,6]];
    /// grid.insert_col(1, vec![9,9]);
    /// assert_eq!(grid[0], [1,9,2,3]);
    /// assert_eq!(grid[1], [4,9,5,6]);
    /// assert_eq!(grid.size(), (2,4))
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - the grid is not empty and `col.len() != grid.rows()`.
    /// - the index is greater than the number of columns
    pub fn insert_col(&mut self, index: usize, col: Vec<T>) {
        let input_len = col.len();
        assert!(
            !(self.rows > 0 && input_len != self.rows),
            "Inserted col must be of length {}, but was {}.",
            self.rows,
            col.len()
        );
        assert!(
            index <= self.cols,
            "Out of range. Index was {}, but must be less or equal to {}.",
            index,
            self.cols
        );
        for (row_iter, col_val) in col.into_iter().enumerate() {
            let data_idx = row_iter * self.cols + index + row_iter;
            self.data.insert(data_idx, col_val);
        }
        self.rows = input_len;
        self.cols += 1;
    }

    /// Returns a reference to the internal data structure of the grid.
    ///
    /// Grid uses a row major layout.
    /// All rows are placed right after each other in the vector data structure.
    ///
    /// # Examples
    /// ```
    /// use grid::*;
    /// let grid = grid![[1,2,3][4,5,6]];
    /// let flat = grid.flatten();
    /// assert_eq!(flat, &vec![1,2,3,4,5,6]);
    /// ```
    #[must_use]
    pub fn flatten(&self) -> &Vec<T> {
        &self.data
    }

    /// Converts self into a vector without clones or allocation.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Transpose the grid so that columns become rows in new grid.
    #[must_use]
    pub fn transpose(&self) -> Grid<T>
    where
        T: Clone,
    {
        let mut data = Vec::with_capacity(self.data.len());
        for c in 0..self.cols {
            for r in 0..self.rows {
                data.push(self[r][c].clone());
            }
        }
        Grid {
            data,
            cols: self.rows,
            rows: self.cols,
        }
    }

    /// Fills the grid with elements by cloning `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use grid::*;
    /// let mut grid = grid![[1,2,3][4,5,6]];
    /// grid.fill(7);
    /// assert_eq!(grid[0], [7,7,7]);
    /// assert_eq!(grid[1], [7,7,7]);
    /// ```
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }

    /// Fills the grid with elements returned by calling a closure repeatedly.
    ///
    /// This method uses a closure to create new values. If you'd rather
    /// [`Clone`] a given value, use [`fill`]. If you want to use the [`Default`]
    /// trait to generate values, you can pass [`Default::default`] as the
    /// argument.
    ///
    /// [`fill`]: Grid::fill
    ///
    /// # Examples
    ///
    /// ```
    /// use grid::*;
    /// let mut grid = grid![[1,2,3][4,5,6]];
    /// grid.fill_with(Default::default);
    /// assert_eq!(grid[0], [0,0,0]);
    /// assert_eq!(grid[1], [0,0,0]);
    /// ```
    pub fn fill_with<F>(&mut self, f: F)
    where
        F: FnMut() -> T,
    {
        self.data.fill_with(f);
    }

    /// Iterate over the rows of the grid. Each time an iterator over a single
    /// row is returned.
    ///
    /// An item in this iterator is equal to a call to `Grid.iter_row(row_index)`
    /// of the corresponding row.
    ///
    /// # Examples
    ///
    /// ```
    /// use grid::*;
    /// let mut grid = grid![[1,2,3][4,5,6]];
    /// let sum_by_row: Vec<u8> = grid.iter_rows().map(|row| row.sum()).collect();
    /// assert_eq!(sum_by_row, vec![1+2+3, 4+5+6])
    /// ```
    #[must_use]
    pub fn iter_rows(&self) -> GridRowIter<'_, T> {
        GridRowIter {
            grid: self,
            row_index: 0,
        }
    }

    /// Iterate over the columns of the grid. Each time an iterator over a single
    /// column is returned.
    ///
    /// An item in this iterator is equal to a call to `Grid.iter_col(col_index)`
    /// of the corresponding column.
    ///
    /// # Examples
    ///
    /// ```
    /// use grid::*;
    /// let mut grid = grid![[1,2,3][4,5,6]];
    /// let sum_by_col: Vec<u8> = grid.iter_cols().map(|col| col.sum()).collect();
    /// assert_eq!(sum_by_col, vec![1+4, 2+5, 3+6])
    /// ```
    #[must_use]
    pub fn iter_cols(&self) -> GridColIter<'_, T> {
        GridColIter {
            grid: self,
            col_index: 0,
        }
    }
}

impl<T: Clone> Clone for Grid<T> {
    fn clone(&self) -> Self {
        Grid {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl<T> Index<usize> for Grid<T> {
    type Output = [T];

    #[inline]
    fn index(&self, idx: usize) -> &[T] {
        let start_idx = idx * self.cols;
        &self.data[start_idx..start_idx + self.cols]
    }
}

impl<T> IndexMut<usize> for Grid<T> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut [T] {
        let start_idx = idx * self.cols;
        &mut self.data[start_idx..start_idx + self.cols]
    }
}

impl<T> Index<(usize, usize)> for Grid<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        assert!(
            !(row >= self.rows || col >= self.cols),
            "grid index out of bounds: ({row},{col}) out of ({},{})",
            self.rows,
            self.cols
        );
        &self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Grid<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(
            !(row >= self.rows || col >= self.cols),
            "grid index out of bounds: ({row},{col}) out of ({},{})",
            self.rows,
            self.cols
        );
        &mut self.data[row * self.cols + col]
    }
}

impl<T: fmt::Debug> fmt::Debug for Grid<T> {
    #[allow(unused_must_use)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[");
        if self.cols > 0 {
            if f.alternate() {
                writeln!(f);
                /*
                    WARNING

                    Compound types becoming enormous as the entire `fmt::Debug` width is applied to each item individually.
                    For tuples and structs define padding and precision arguments manually to improve readability.
                */
                let width = f.width().unwrap_or(
                    /*
                        Conditionally calculate the longest item by default.
                    */
                    self.data
                        .iter()
                        .map(|i| format!("{i:?}").len())
                        .max()
                        .unwrap(),
                );
                let precision = f.precision().unwrap_or(2);
                for (i, _) in self.data.iter().enumerate().step_by(self.cols) {
                    let mut row = self.data[i..(i + self.cols)].iter().peekable();
                    write!(f, "    [");
                    while let Some(item) = row.next() {
                        write!(
                            f,
                            " {item:width$.precision$?}",
                            // width = width,
                            // precision = precision
                        );
                        if row.peek().is_some() {
                            write!(f, ",");
                        }
                    }
                    writeln!(f, "]");
                }
            } else {
                for (i, _) in self.data.iter().enumerate().step_by(self.cols) {
                    write!(f, "{:?}", &self.data[i..(i + self.cols)]);
                }
            }
        }
        write!(f, "]")
    }
}

impl<T: Eq> PartialEq for Grid<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl<T: Eq> Eq for Grid<T> {}

pub struct GridRowIter<'a, T> {
    grid: &'a Grid<T>,
    row_index: usize,
}
pub struct GridColIter<'a, T> {
    grid: &'a Grid<T>,
    col_index: usize,
}

pub struct GridIndexedIter<'a, T> {
    grid: &'a Grid<T>,
    index: usize
}

impl<'a, T> Iterator for GridRowIter<'a, T> {
    type Item = Iter<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let rows = self.grid.rows();
        let row_index = self.row_index;

        if !(0..rows).contains(&row_index) {
            return None;
        }

        let row_iter = self.grid.iter_row(row_index);
        self.row_index += 1;
        Some(row_iter)
    }
}

impl<'a, T> Iterator for GridColIter<'a, T> {
    type Item = StepBy<Iter<'a, T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let cols = self.grid.cols();
        let col_index = self.col_index;

        if !(0..cols).contains(&col_index) {
            return None;
        }

        let row_iter = self.grid.iter_col(col_index);
        self.col_index += 1;
        Some(row_iter)
    }
}

impl<'a, T> Iterator for GridIndexedIter<'a, T> {
    type Item = ((usize, usize), &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.grid.cols * self.grid.rows {
            return None;
        }

        let row = self.index / self.grid.cols;
        let col = self.index % self.grid.cols;
        let item = &self.grid.data[self.index];

        self.index += 1;
        
        Some(((row, col), item))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[cfg(all(not(feature = "std")))]
    use alloc::string::String;

    #[test]
    fn from_vec_zero_with_cols() {
        let grid: Grid<u8> = Grid::from_vec(vec![], 1);
        assert_eq!(grid.rows(), 0);
        assert_eq!(grid.cols(), 0);
    }

    #[test]
    fn from_vec_zero() {
        let grid: Grid<u8> = Grid::from_vec(vec![], 0);
        let _ = grid.is_empty();
        assert_eq!(grid.rows(), 0);
        assert_eq!(grid.cols(), 0);
    }

    #[test]
    fn insert_col_at_end() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3, 4], 2);
        grid.insert_col(2, vec![5, 6]);
        assert_eq!(grid[0], [1, 2, 5]);
        assert_eq!(grid[1], [3, 4, 6]);
    }

    #[test]
    #[should_panic]
    fn insert_col_out_of_idx() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3, 4], 2);
        grid.insert_col(3, vec![4, 5]);
    }

    #[test]
    fn insert_row_at_end() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3, 4], 2);
        grid.insert_row(2, vec![5, 6]);
        assert_eq!(grid[0], [1, 2]);
        assert_eq!(grid[1], [3, 4]);
        assert_eq!(grid[2], [5, 6]);
    }

    #[test]
    fn insert_row_empty() {
        let mut grid: Grid<u8> = grid![];
        grid.insert_row(0, vec![1, 2, 3]);
        assert_eq!(grid[0], [1, 2, 3]);
        assert_eq!((1, 3), grid.size());
    }

    #[test]
    fn insert_col_empty() {
        let mut grid: Grid<u8> = grid![];
        grid.insert_col(0, vec![1, 2, 3]);
        assert_eq!(grid[0], [1]);
        assert_eq!((3, 1), grid.size());
    }

    #[test]
    #[should_panic]
    fn insert_row_out_of_idx() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3, 4], 2);
        grid.insert_row(3, vec![4, 5]);
    }

    #[test]
    #[should_panic]
    fn insert_row_wrong_size_of_idx() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3, 4], 2);
        grid.insert_row(1, vec![4, 5, 4]);
    }

    #[test]
    fn insert_row_start() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3, 4], 2);
        let new_row = [5, 6];
        grid.insert_row(1, new_row.to_vec());
        assert_eq!(grid[1], new_row);
    }

    #[test]
    fn pop_col_1x3() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3], 3);
        assert_eq!(grid.pop_col(), Some(vec![3]));
        assert_eq!(grid.size(), (1, 2));
        assert_eq!(grid.pop_col(), Some(vec![2]));
        assert_eq!(grid.size(), (1, 1));
        assert_eq!(grid.pop_col(), Some(vec![1]));
        assert!(grid.is_empty());
        assert_eq!(grid.pop_col(), None);
    }

    #[test]
    fn pop_col_3x1() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3], 1);
        assert_eq!(grid.pop_col(), Some(vec![1, 2, 3]));
        assert!(grid.is_empty());
        assert_eq!(grid.pop_col(), None);
    }

    #[test]
    fn pop_col_2x2() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3, 4], 2);
        assert_eq!(grid.pop_col(), Some(vec![2, 4]));
        assert_eq!(grid.size(), (2, 1));
        assert_eq!(grid.pop_col(), Some(vec![1, 3]));
        assert_eq!(grid.size(), (0, 0));
        assert_eq!(grid.pop_col(), None);
    }

    #[test]
    fn pop_col_3x4() {
        let mut grid: Grid<u16> =
            Grid::from_vec(vec![1, 2, 3, 4, 11, 22, 33, 44, 111, 222, 333, 444], 4);
        assert_eq!(grid.pop_col(), Some(vec![4, 44, 444]));
        assert_eq!(grid.size(), (3, 3));
        assert_eq!(grid.pop_col(), Some(vec![3, 33, 333]));
        assert_eq!(grid.size(), (3, 2));
        assert_eq!(grid.pop_col(), Some(vec![2, 22, 222]));
        assert_eq!(grid.size(), (3, 1));
        assert_eq!(grid.pop_col(), Some(vec![1, 11, 111]));
        assert_eq!(grid.size(), (0, 0));
        assert_eq!(grid.pop_col(), None);
    }

    #[test]
    fn pop_col_empty() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![], 0);
        assert_eq!(grid.pop_row(), None);
    }

    #[test]
    fn pop_row_2x2() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3, 4], 2);
        assert_eq!(grid.pop_row(), Some(vec![3, 4]));
        assert_ne!(grid.size(), (1, 4));
        assert_eq!(grid.pop_row(), Some(vec![1, 2]));
        assert_eq!(grid.size(), (0, 0));
        assert_eq!(grid.pop_row(), None);
    }

    #[test]
    fn pop_row_empty() {
        let mut grid: Grid<u8> = Grid::from_vec(vec![], 0);
        assert_eq!(grid.pop_row(), None);
    }

    #[test]
    fn ne_full_empty() {
        let g1 = Grid::from_vec(vec![1, 2, 3, 4], 2);
        let g2: Grid<u8> = grid![];
        assert_ne!(g1, g2);
    }

    #[test]
    fn ne() {
        let g1 = Grid::from_vec(vec![1, 2, 3, 5], 2);
        let g2 = Grid::from_vec(vec![1, 2, 3, 4], 2);
        assert_ne!(g1, g2);
    }

    #[test]
    fn ne_dif_rows() {
        let g1 = Grid::from_vec(vec![1, 2, 3, 4], 2);
        let g2 = Grid::from_vec(vec![1, 2, 3, 4], 1);
        assert_ne!(g1, g2);
    }

    #[test]
    fn equal_empty() {
        let grid: Grid<char> = grid![];
        let grid2: Grid<char> = grid![];
        assert_eq!(grid, grid2);
    }
    #[test]
    fn equal() {
        let grid: Grid<char> = grid![['a', 'b', 'c', 'd']['a', 'b', 'c', 'd']['a', 'b', 'c', 'd']];
        let grid2: Grid<char> = grid![['a', 'b', 'c', 'd']['a', 'b', 'c', 'd']['a', 'b', 'c', 'd']];
        assert_eq!(grid, grid2);
    }

    #[test]
    #[should_panic]
    fn idx_out_of_col_bounds() {
        let grid: Grid<char> = grid![['a', 'b', 'c', 'd']['a', 'b', 'c', 'd']['a', 'b', 'c', 'd']];
        let _ = grid[0][5];
    }

    #[test]
    #[should_panic]
    fn idx_tup_out_of_col_bounds() {
        let grid: Grid<char> = grid![['a', 'b', 'c', 'd']['a', 'b', 'c', 'd']['a', 'b', 'c', 'd']];
        let _ = grid[(0, 5)];
    }

    #[test]
    fn push_col_2x3() {
        let mut grid: Grid<u8> = grid![  
                    [0, 1, 2]
                    [10, 11, 12]];
        grid.push_col(vec![3, 13]);
        assert_eq!(grid.size(), (2, 4));
        assert_eq!(
            grid.iter_row(0).copied().collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            grid.iter_row(1).copied().collect::<Vec<_>>(),
            vec![10, 11, 12, 13]
        );
    }

    #[test]
    fn push_col_3x4() {
        let mut grid: Grid<char> = grid![  
                    ['a', 'b', 'c', 'd']
                    ['a', 'b', 'c', 'd']
                    ['a', 'b', 'c', 'd']];
        grid.push_col(vec!['x', 'y', 'z']);
        assert_eq!(grid.size(), (3, 5));
        assert_eq!(
            grid.iter_row(0).copied().collect::<Vec<_>>(),
            vec!['a', 'b', 'c', 'd', 'x']
        );
        assert_eq!(
            grid.iter_row(1).copied().collect::<Vec<_>>(),
            vec!['a', 'b', 'c', 'd', 'y']
        );
        assert_eq!(
            grid.iter_row(2).copied().collect::<Vec<_>>(),
            vec!['a', 'b', 'c', 'd', 'z']
        );
    }

    #[test]
    fn push_col_1x3() {
        let mut grid: Grid<char> = grid![['a', 'b', 'c']];
        grid.push_col(vec!['d']);
        assert_eq!(grid.size(), (1, 4));
        assert_eq!(grid[0][3], 'd');
    }

    #[test]
    fn push_col_empty() {
        let mut grid: Grid<char> = grid![];
        grid.push_col(vec!['b', 'b', 'b', 'b']);
        assert_eq!(grid.size(), (4, 1));
        assert_eq!(grid[0][0], 'b');
    }

    #[test]
    #[should_panic]
    fn push_col_wrong_size() {
        let mut grid: Grid<char> = grid![['a','a','a']['a','a','a']];
        grid.push_col(vec!['b']);
        grid.push_col(vec!['b', 'b']);
    }

    #[test]
    #[should_panic]
    fn push_col_zero_len() {
        let mut grid: Grid<char> = grid![];
        grid.push_col(vec![]);
    }

    #[test]
    fn push_row_empty() {
        let mut grid: Grid<char> = grid![];
        grid.push_row(vec!['b', 'b', 'b', 'b']);
        assert_eq!(grid.size(), (1, 4));
        assert_eq!(grid[0][0], 'b');
    }

    #[test]
    #[should_panic]
    fn push_empty_row() {
        let mut grid = Grid::init(0, 1, 0);
        grid.push_row(vec![]);
    }

    #[test]
    #[should_panic]
    fn push_row_wrong_size() {
        let mut grid: Grid<char> = grid![['a','a','a']['a','a','a']];
        grid.push_row(vec!['b']);
        grid.push_row(vec!['b', 'b', 'b', 'b']);
    }

    #[test]
    fn iter_row() {
        let grid: Grid<u8> = grid![[1,2,3][1,2,3]];
        let mut iter = grid.iter_row(0);
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[should_panic]
    fn iter_row_empty() {
        let grid: Grid<u8> = grid![];
        let _ = grid.iter_row(0);
    }

    #[test]
    #[should_panic]
    fn iter_row_out_of_bound() {
        let grid: Grid<u8> = grid![[1,2,3][1,2,3]];
        let _ = grid.iter_row(2);
    }

    #[test]
    #[should_panic]
    fn iter_col_out_of_bound() {
        let grid: Grid<u8> = grid![[1,2,3][1,2,3]];
        let _ = grid.iter_col(3);
    }

    #[test]
    #[should_panic]
    fn iter_col_zero() {
        let grid: Grid<u8> = grid![];
        let _ = grid.iter_col(0);
    }

    #[test]
    fn iter() {
        let grid: Grid<u8> = grid![[1,2][3,4]];
        let mut iter = grid.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn indexed_iter() {
        let grid: Grid<u8> = grid![[1,2][3,4]];
        let mut iter: GridIndexedIter<u8> = grid.indexed_iter();
        assert_eq!(iter.next(), Some(((0, 0), &1)));
        assert_eq!(iter.next(), Some(((0, 1), &2)));
        assert_eq!(iter.next(), Some(((1, 0), &3)));
        assert_eq!(iter.next(), Some(((1, 1), &4)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn clear() {
        let mut grid: Grid<u8> = grid![[1, 2, 3]];
        assert!(!grid.is_empty());
        grid.clear();
        assert!(grid.is_empty());
    }

    #[test]
    fn is_empty_false() {
        let grid: Grid<u8> = grid![[1, 2, 3]];
        assert!(!grid.is_empty());
    }

    #[test]
    fn is_empty() {
        let mut g: Grid<u8> = grid![[]];
        assert!(g.is_empty());
        g = grid![];
        assert!(g.is_empty());
        g = Grid::from_vec(vec![], 0);
        assert!(g.is_empty());
        g = Grid::new(0, 0);
        assert!(g.is_empty());
        g = Grid::new(0, 1);
        assert!(g.is_empty());
        g = Grid::new(1, 0);
        assert!(g.is_empty());
        g = Grid::init(0, 0, 10);
        assert!(g.is_empty());
    }

    #[test]
    fn fmt_empty() {
        let grid: Grid<u8> = grid![];
        assert_eq!(format!("{:?}", grid), "[]");
    }

    #[test]
    fn fmt_row() {
        let grid: Grid<u8> = grid![[1, 2, 3]];
        assert_eq!(format!("{:?}", grid), "[[1, 2, 3]]");
    }

    #[test]
    fn fmt_grid() {
        let grid: Grid<u8> = grid![[1,2,3][4,5,6][7,8,9]];
        assert_eq!(format!("{:?}", grid), "[[1, 2, 3][4, 5, 6][7, 8, 9]]");
    }

    #[test]
    fn fmt_pretty_empty() {
        let grid: Grid<f32> = grid![];
        assert_eq!(format!("{:#?}", grid), "[]");
    }

    #[test]
    fn fmt_pretty_int() {
        let grid: Grid<u8> = grid![
            [1,2,3]
            [4,5,6]
            [7,8,95]
        ];

        let expected_output = r#"[
    [  1,  2,  3]
    [  4,  5,  6]
    [  7,  8, 95]
]"#;

        assert_eq!(format!("{:#?}", grid), expected_output);

        let expected_output = r#"[
    [   1,   2,   3]
    [   4,   5,   6]
    [   7,   8,  95]
]"#;

        assert_eq!(format!("{:#3?}", grid), expected_output);
    }

    #[test]
    fn fmt_pretty_float() {
        let grid: Grid<f32> = grid![
            [1.5,2.6,3.44]
            [4.775,5.,6.]
            [7.1,8.23444,95.55]
        ];

        let expected_output = r#"[
    [   1.5,   2.6,   3.4]
    [   4.8,   5.0,   6.0]
    [   7.1,   8.2,  95.6]
]"#;

        assert_eq!(format!("{:#5.1?}", grid), expected_output);

        let expected_output = r#"[
    [  1.50000,  2.60000,  3.44000]
    [  4.77500,  5.00000,  6.00000]
    [  7.10000,  8.23444, 95.55000]
]"#;

        assert_eq!(format!("{:#8.5?}", grid), expected_output);
    }

    #[test]
    fn fmt_pretty_tuple() {
        let grid: Grid<(i32, i32)> = grid![
            [(5,66), (432, 55)]
            [(80, 90), (5, 6)]
        ];

        let expected_output = r#"[
    [ (        5,        66), (      432,        55)]
    [ (       80,        90), (        5,         6)]
]"#;

        assert_eq!(format!("{grid:#?}"), expected_output);

        let expected_output = r#"[
    [ (  5,  66), (432,  55)]
    [ ( 80,  90), (  5,   6)]
]"#;

        assert_eq!(format!("{:#3?}", grid), expected_output);
    }

    #[test]
    fn fmt_pretty_struct_derived() {
        #[derive(Debug)]
        struct Person {
            _name: String,
            _precise_age: f32,
        }

        impl Person {
            fn new(name: &str, precise_age: f32) -> Self {
                Person {
                    _name: name.into(),
                    _precise_age: precise_age,
                }
            }
        }

        let grid: Grid<Person> = grid![
            [Person::new("Vic", 24.5), Person::new("Mr. Very Long Name", 1955.)]
            [Person::new("Sam", 8.9995), Person::new("John Doe", 40.14)]
        ];

        let expected_output = r#"[
    [ Person { _name: "Vic", _precise_age: 24.50000 }, Person { _name: "Mr. Very Long Name", _precise_age: 1955.00000 }]
    [ Person { _name: "Sam", _precise_age: 8.99950 }, Person { _name: "John Doe", _precise_age: 40.14000 }]
]"#;

        assert_eq!(format!("{:#5.5?}", grid), expected_output);
    }

    #[test]
    fn clone() {
        let grid = grid![[1, 2, 3][4, 5, 6]];
        let mut clone = grid.clone();
        clone[0][2] = 10;
        assert_eq!(grid[0][2], 3);
        assert_eq!(clone[0][2], 10);
    }

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
    #[should_panic]
    fn from_vec_panics_1() {
        let _: Grid<u8> = Grid::from_vec(vec![1, 2, 3], 0);
    }

    #[test]
    #[should_panic]
    fn from_vec_panics_2() {
        let _: Grid<u8> = Grid::from_vec(vec![1, 2, 3], 2);
    }

    #[test]
    fn from_vec_uses_original_vec() {
        let capacity = 10_000_000;
        let vec = Vec::with_capacity(capacity);
        let grid: Grid<u8> = Grid::from_vec(vec, 0);
        assert!(grid.into_vec().capacity() >= capacity);
    }

    #[test]
    fn init() {
        Grid::init(1, 2, 3);
        Grid::init(1, 2, 1.2);
        Grid::init(1, 2, 'a');
    }

    #[test]
    fn init_empty() {
        let grid = Grid::init(0, 1, 0);
        assert!(grid.is_empty());
        assert_eq!(grid.cols(), 0);
        assert_eq!(grid.rows(), 0);
    }

    #[test]
    fn new() {
        let grid: Grid<u8> = Grid::new(1, 2);
        assert_eq!(grid[0][0], 0);
    }

    #[test]
    #[should_panic]
    fn new_panics() {
        let _: Grid<u8> = Grid::new(usize::MAX, 2);
    }

    #[test]
    fn new_empty() {
        let grid: Grid<u8> = Grid::new(0, 1);
        assert!(grid.is_empty());
        assert_eq!(grid.cols(), 0);
        assert_eq!(grid.rows(), 0);
    }

    #[test]
    #[should_panic]
    fn init_panics() {
        Grid::init(usize::MAX, 2, 3);
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
    fn get_mut() {
        let mut grid = Grid::init(1, 2, 3);
        let mut_ref = grid.get_mut(0, 0).unwrap();
        *mut_ref = 5;
        assert_eq!(grid[0][0], 5);
    }

    #[test]
    fn get_mut_none() {
        let mut grid = Grid::init(1, 2, 3);
        let mut_ref = grid.get_mut(1, 4);
        assert_eq!(mut_ref, None);
    }

    #[test]
    fn idx() {
        let grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3, 4], 2);
        assert_eq!(grid[0][0], 1);
        assert_eq!(grid[0][1], 2);
        assert_eq!(grid[1][0], 3);
        assert_eq!(grid[1][1], 4);
    }

    #[test]
    fn idx_tup() {
        let grid: Grid<u8> = Grid::from_vec(vec![1, 2, 3, 4], 2);
        assert_eq!(grid[(0, 0)], 1);
        assert_eq!(grid[(0, 1)], 2);
        assert_eq!(grid[(1, 0)], 3);
        assert_eq!(grid[(1, 1)], 4);
    }

    #[test]
    #[should_panic]
    fn idx_panic_1() {
        let grid = Grid::init(1, 2, 3);
        let _ = grid[20][0];
    }

    #[test]
    #[should_panic]
    fn idx_tup_panic_1() {
        let grid = Grid::init(1, 2, 3);
        let _ = grid[(20, 0)];
    }

    #[test]
    #[should_panic]
    fn idx_panic_2() {
        let grid = Grid::init(1, 2, 3);
        let _ = grid[0][20];
    }

    #[test]
    #[should_panic]
    fn idx_tup_panic_2() {
        let grid = Grid::init(1, 2, 3);
        let _ = grid[(0, 20)];
    }

    #[test]
    fn idx_set() {
        let mut grid = Grid::init(1, 2, 3);
        grid[0][0] = 4;
        assert_eq!(grid[0][0], 4);
    }

    #[test]
    fn idx_tup_set() {
        let mut grid = Grid::init(1, 2, 3);
        grid[(0, 0)] = 4;
        assert_eq!(grid[(0, 0)], 4);
    }

    #[test]
    fn size() {
        let grid = Grid::init(1, 2, 3);
        assert_eq!(grid.size(), (1, 2));
    }

    #[test]
    fn transpose() {
        let grid: Grid<u8> = grid![[1,2,3][4,5,6]];
        assert_eq!(format!("{:?}", grid.transpose()), "[[1, 4][2, 5][3, 6]]");
    }

    #[test]
    fn fill() {
        let mut grid: Grid<u8> = grid![[1,2,3][4,5,6]];
        grid.fill(7);
        assert_eq!(grid[0], [7, 7, 7]);
        assert_eq!(grid[1], [7, 7, 7]);
    }

    #[test]
    fn fill_with() {
        let mut grid: Grid<u8> = grid![[1,2,3][4,5,6]];
        grid.fill_with(Default::default);
        assert_eq!(grid[0], [0, 0, 0]);
        assert_eq!(grid[1], [0, 0, 0]);
    }

    #[test]
    fn iter_rows() {
        let grid: Grid<u8> = grid![[1,2,3][4,5,6]];
        let max_by_row: Vec<u8> = grid
            .iter_rows()
            .map(|row| row.max().unwrap())
            .copied()
            .collect();
        assert_eq!(max_by_row, vec![3, 6]);

        let sum_by_row: Vec<u8> = grid.iter_rows().map(|row| row.sum()).collect();
        assert_eq!(sum_by_row, vec![1 + 2 + 3, 4 + 5 + 6]);
    }

    #[test]
    fn iter_cols() {
        let grid: Grid<u8> = grid![[1,2,3][4,5,6]];
        let max_by_col: Vec<u8> = grid
            .iter_cols()
            .map(|col| col.max().unwrap())
            .copied()
            .collect();

        assert_eq!(max_by_col, vec![4, 5, 6]);

        let sum_by_col: Vec<u8> = grid.iter_cols().map(|col| col.sum()).collect();
        assert_eq!(sum_by_col, vec![1 + 4, 2 + 5, 3 + 6]);
    }
    #[test]
    fn remove_col() {
        let mut grid = grid![[1,2,3,4][5,6,7,8][9,10,11,12][13,14,15,16]];
        assert_eq![grid.remove_col(3), Some(vec![4, 8, 12, 16])];
        assert_eq![grid.remove_col(0), Some(vec![1, 5, 9, 13])];
        assert_eq![grid.remove_col(1), Some(vec![3, 7, 11, 15])];
        assert_eq![grid.remove_col(0), Some(vec![2, 6, 10, 14])];
        assert_eq![grid.remove_col(0), None];
    }
    #[test]
    fn remove_row() {
        let mut grid = grid![[1,2][3,4][5,6]];
        assert_eq![grid.remove_row(1), Some(vec![3, 4])];
        assert_eq![grid.remove_row(0), Some(vec![1, 2])];
        assert_eq![grid.remove_row(0), Some(vec![5, 6])];
        assert_eq![grid.remove_row(0), None];
    }
    #[test]
    fn remove_row_out_of_bound() {
        let mut grid = grid![[1, 2][3, 4]];
        assert_eq![grid.remove_row(5), None];
        assert_eq![grid.remove_row(1), Some(vec![3, 4])];
    }
    #[test]
    fn remove_col_out_of_bound() {
        let mut grid = grid![[1, 2][3, 4]];
        assert_eq!(grid.remove_col(5), None);
        assert_eq!(grid.remove_col(1), Some(vec![2, 4]));
    }
}
