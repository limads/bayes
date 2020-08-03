use nalgebra::*;
use nalgebra::storage::*;

/// Structure that encapsulates the logic of running windows over a matrix.
/// It holds information such as the window size, the step,
/// and whether the ordering is row-wise or column-wise. The user only uses
/// the specializations of this structure returned by WindowIterate::windows()
/// and ChunkIterate::chunks() that are implemented for DMatrix<N>, that
/// pick a step size at compile time (for contiguous overlapping windows in the first case
/// and contiguous non-overlapping windows in the second case). The current definition always
/// slides over a matrix in the row direction first.
pub struct PatchIterator<'a, N, S, W, V>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>,
        S : Storage<N, Dynamic, Dynamic>,
        W : Dim,
        V : Dim
{
    source : &'a Matrix<N, Dynamic, Dynamic, S>,
    size : (usize, usize),

    /// Patch iterator differs from a standard Vec iterator
    /// simply by having a internal counter over two dimensions:
    /// the first count the current rows and the last count the
    /// current column. Different implementations of patch iterator
    /// just differ by the rule by which those quantities evolve at each
    /// iteration.
    curr_pos : (usize, usize),

    /// Unused for now.
    _c_stride : usize,

    /// Vertical increment. Either U1 or Dynamic.
    step_v : V,

    /// Horizontal increment. Either U1 or Dynamic.
    step_h : W,

    /// Unused for now. Assume row-wise.
    _row_wise : bool,

    /// A pooling operation very often follows iterating over a matrix.
    /// pool_dims keep the dimensions of the resulting pooling, so the
    /// pool method can be generic, while the pooling calculation logic
    /// is different for each implementation.
    pool_dims : (usize, usize)
}

impl<'a, N, S, W, V> PatchIterator<'a, N, S, W, V>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>,
        S : Storage<N, Dynamic, Dynamic>,
        W : Dim,
        V : Dim
{

    /// PatchIterator::pool consumes the structure, iterate over its source matrix
    /// and generates another owned matrix by applying any function that consumes a matrix
    /// and returns a scalar, such as max(), sum() and norm(). If the original matrix has
    /// dimensions (r,c) and the window size asked by the user has dimensions (wr, wc),
    /// the resulting matrix has dimensions (r/wr, c/wc).
    pub fn pool<F>(
        mut self,
        mut f : F
    ) -> Matrix<N, Dynamic, Dynamic, VecStorage<N, Dynamic, Dynamic>>
        where
            F : FnMut(Matrix<N, Dynamic, Dynamic, SliceStorage<'a, N, Dynamic, Dynamic, S::RStride, S::CStride>>)->N
    {
        //println!("Pool dims: {:?}", self.pool_dims);
        let mut data : Vec<N> = Vec::with_capacity(self.pool_dims.1 * self.pool_dims.0);

        // Since iteration is row-wise, we need to transpose the matrix.
        while let Some(w) = self.next() {
            let s = f(w);
            data.push(s);
        }
        //println!("Data size: {:?}", data.len());
        let mut ans = DMatrix::<N>::from_vec(self.pool_dims.1, self.pool_dims.0, data);
        ans.transpose_mut();
        ans
    }
}

/// Returns a WindowIterator over overlapping contiguous regions with step size 1
pub trait WindowIterate<N, S>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>
        S : Storage<N, Dynamic, Dynamic>,
{
    fn windows(&self, win_sz : (usize, usize)) -> PatchIterator<N, S, U1, U1>;
}

/// Returns a WindowIterator over non-overlapping contiguous regions
pub trait ChunkIterate<N, S>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>,
        S : Storage<N, Dynamic, Dynamic>,
{
    fn chunks(&self, sz : (usize, usize)) -> PatchIterator<N, S, Dynamic, Dynamic>;

}

impl<'a, N, S, W, V> Iterator for PatchIterator<'a, N, S, W, V>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>,
        S : Storage<N, Dynamic, Dynamic>,
        W : Dim,
        V : Dim
    {

    type Item = Matrix<N, Dynamic, Dynamic, SliceStorage<'a, N, Dynamic, Dynamic, S::RStride, S::CStride>>;

    fn next(&mut self) -> Option<Self::Item> {
        let win = if self.curr_pos.0  + self.size.0 <= self.source.nrows() && self.curr_pos.1 + self.size.1 <= self.source.ncols() {
            //println!("Matrix slice: pos : {:?} slice : {:?} size: {:?}", self.curr_pos, self.size, self.source.shape());
            Some(self.source.slice(self.curr_pos, self.size))
        } else {
            None
        };
        self.curr_pos.1 += self.step_h.value(); // self.size.1 for chunks; 1 for window
        if self.curr_pos.1 + self.size.1 > self.source.ncols() { // >=
            self.curr_pos.1 = 0;
            self.curr_pos.0 += self.step_v.value();
        }
        win
    }

}

impl<N, S> WindowIterate<N, S> for Matrix<N, Dynamic, Dynamic, S>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>
        S : Storage<N, Dynamic, Dynamic>,
{
    fn windows(
        &self,
        sz : (usize, usize)
    ) -> PatchIterator<N, S, U1, U1> {
        if self.nrows() % sz.0 != 0 || self.ncols() % sz.1 != 0 {
            panic!("Matrix size should be a multiple of window size");
        }
        let pool_dims = (self.nrows() - sz.0 + 1, self.ncols() - sz.1 + 1);
        PatchIterator::<N, S, U1, U1> {
            source : &self,
            size : sz,
            curr_pos : (0, 0),
            _c_stride : self.nrows(),
            step_h : U1{},
            step_v : U1{},
            _row_wise : false,
            pool_dims
        }
    }
}

impl<N, S> ChunkIterate<N, S> for Matrix<N, Dynamic, Dynamic, S>
    where
        N : Scalar,
        //S : ContiguousStorage<N, Dynamic, Dynamic>
        S : Storage<N, Dynamic, Dynamic>,
{
    fn chunks(
        &self,
        sz : (usize, usize)
    ) -> PatchIterator<N, S, Dynamic, Dynamic> {
        let step_v = Dim::from_usize(sz.0);
        let step_h = Dim::from_usize(sz.1);
        //println!("matrix size: {:?}; window size: {:?}", self.shape(), sz);
        if self.nrows() % sz.0 != 0 || self.ncols() % sz.1 != 0 {
            panic!("Matrix size should be a multiple of window size");
        }
        let pool_dims = (self.nrows() / sz.0, self.ncols() / sz.1);
        PatchIterator::<N, S, Dynamic, Dynamic> {
            source : &self,
            size : sz,
            curr_pos : (0, 0),
            _c_stride : self.nrows(),
            step_v,
            step_h,
            _row_wise : false,
            pool_dims
        }
    }
}

