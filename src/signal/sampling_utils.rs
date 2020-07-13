use nalgebra::{Scalar, DMatrix, Dynamic, U1, MatrixSlice, MatrixSliceMut};
use std::ops::{Add, Mul};

#[inline(always)]
fn zip_rows_buffer_mut<'a, N : Scalar>(
    buf : &'a mut [u8],
    dst : &'a DMatrix<N>,
    c_stride : usize,
    skip : (usize, usize)
) -> impl Iterator<Item = (&'a mut [u8], MatrixSlice<'a, N, U1, Dynamic, U1, Dynamic>)> {
    let buf_rows = buf.chunks_mut(c_stride)
        .skip(skip.0)
        .take(dst.nrows())
        .map(move |r| &mut r[(skip.1)..(skip.1+dst.ncols())]);
    let dst_rows = dst.row_iter();
    buf_rows.zip(dst_rows)
}

#[inline(always)]
fn zip_rows_dst_mut<'a, N : Scalar>(
    buf : &'a [u8],
    dst : &'a mut DMatrix<N>,
    c_stride : usize,
    skip : (usize, usize)
) -> impl Iterator<Item = (&'a [u8], MatrixSliceMut<'a, N, U1, Dynamic, U1, Dynamic>)> {
    let win_cols = dst.ncols();
    let buf_rows = buf.chunks(c_stride)
        .skip(skip.0)
        .take(dst.nrows())
        .map(move |r| &r[(skip.1)..(skip.1+win_cols)]);
    let dst_rows = dst.row_iter_mut();
    buf_rows.zip(dst_rows)
}

fn read_from<N>(
    src : &mut DMatrix<N>,
    buf : &[u8],
    start : (usize, usize),
    buf_dims : (usize, usize),
    offset : Option<N>,
    scale : Option<N>
) -> ()
where
    N : Scalar + From<f32> + From<u8> + Add<Output=N> + Mul<Output=N> + Copy
{
    let offset = offset.unwrap_or(N::from(0.0));
    let scale = scale.unwrap_or(N::from(1.0));
    if check_bounds(start, src.shape(), buf_dims) {
        for (buf_row, mut dst_row) in zip_rows_dst_mut(buf, src, buf_dims.1, start) {
            buf_row.iter()
                .zip(dst_row.iter_mut())
                .for_each(|(b, d)|{
                    *d = (N::from(*b)+offset)*scale;
                })
        }
    } else {
        println!("[update_from] Out of buffer bounds. Skipping.");
    }
}

#[inline(always)]
fn write_into<N>(
    src : &DMatrix<N>,
    buf : &mut [u8],
    start : (usize, usize),
    buf_dims : (usize, usize),
    offset : Option<N>,
    scale : Option<N>
) ->()
where
    N : Scalar + Into<u8> + Add<Output=N> + Mul<Output=N> + From<f32> + Copy,
    u8 : From<N>
{
    let offset = offset.unwrap_or(N::from(0.0));
    let scale = scale.unwrap_or(N::from(1.0));
    if check_bounds(start, src.shape(), buf_dims) {
        for (buf_row, dst_row) in zip_rows_buffer_mut(buf, &src, buf_dims.1, start) {
        buf_row.iter_mut()
            .zip(dst_row.iter())
            .for_each(|(b, d)|{
                *b = u8::from((*d + offset) * scale)
            })
        }
    } else {
        println!("[write_into] Out of buffer bounds. Skipping.");
    }
}

// From trait cannot be used for f32/f64->u8 casting, which is why
// we need to do cast to a f64 first then cast to a u8 (this lossy cast
// can only be done with concerete types). The alternative would be to
// cast into f32, but then we could not implement the Signal<T> trait in
// a generic way for floating-point containers.
pub fn write_matrix_to_slice<N>(
    dm : &DMatrix<N>,
    buf : &mut [u8],
    buf_dims : (usize, usize),
    start : (usize, usize),
    offset : Option<N>,
    scale : Option<N>
) -> ()
where
    N : Scalar + Add<Output=N> + Mul<Output=N> + From<f32> + Copy,
    f64 : From<N>
{
    let rows = buf.chunks_mut(buf_dims.1).skip(start.0).take(dm.nrows());
    let offset = offset.unwrap_or(N::from(0.0));
    let scale = scale.unwrap_or(N::from(1.0));
    if start.0 + dm.nrows() <= buf_dims.0 && start.1 + dm.ncols() <= buf_dims.1 {
        for (i, r) in rows.enumerate() {
            r[(start.1)..(start.1 + dm.ncols())]
                .iter_mut()
                .zip(dm.row(i).iter())
                .for_each(|(b, f)|{
                    *b = f64::from(((*f + offset) * scale)) as u8;
                });
        }
    } else {
        println!("Out of bounds for drawing matrix");
    }
}

#[inline(always)]
pub fn check_bounds(
    start : (usize, usize),
    win_dims : (usize, usize),
    buf_dims : (usize, usize)
) -> bool {
    start.0 + win_dims.0 <= buf_dims.0 && start.1 + win_dims.1 <= buf_dims.1
}

#[inline(always)]
pub fn convert_slice<N>(src : &[u8], dst : &mut [N])
where
    N : Scalar + From<u8>
{
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = N::from(*s);
    }
}

/*#[inline(always)]
pub fn convert_f32_slice(src : &[u8], dst : &mut [f32]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as f32;
    }
}

#[inline(always)]
pub fn convert_f64_slice(src : &[u8], dst : &mut [f64]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as f64;
    }
}*/

#[inline(always)]
pub fn convert_slice_strided<N>(src : &[u8], dst : &mut [N], cstride : usize)
where
    N : Scalar + From<u8>
{
    for i in 0..dst.len() {
        dst[i] = N::from(src[i*cstride])
    }
}

/*#[inline(always)]
pub fn convert_f32_slice_strided(src : &[u8], dst : &mut [f32], cstride : usize) {
    for i in 0..dst.len() {
        dst[i] = src[i*cstride] as f32
    }
}

#[inline(always)]
pub fn convert_f64_slice_strided(src : &[u8], dst : &mut [f64], cstride : usize) {
    for i in 0..dst.len() {
        dst[i] = src[i*cstride] as f64
    }
}*/

#[inline(always)]
pub fn copy_from_slices<N>(d : &mut DMatrix<N>, data : &[&[u8]], step : usize)
where
    N : Scalar + From<u8>
{
    assert!(data.len() > 1);
    let sz = data[0].len();
    assert!(sz / step == d.ncols());
    //println!("row slices passed = {}", data.len());
    //println!("current size = {}", d.nrows());
    assert!(data.len() == d.nrows());
    for d in data.iter().skip(1) {
        assert!(d.len() == sz);
    }
    for (i, mut row) in d.row_iter_mut().enumerate() {
        for (j, e) in row.iter_mut().enumerate() {
            *e = N::from(data[i][j*step]);
        }
    }
}

/*#[inline(always)]
pub fn copy_from_slices_f32(d : &mut DMatrix<f32>, data : &[&[u8]], step : usize) {
    assert!(data.len() > 1);
    let sz = data[0].len();
    assert!(sz / step == d.ncols());
    assert!(data.len() == d.nrows());
    for d in data.iter().skip(1) {
        assert!(d.len() == sz);
    }
    for (i, mut row) in d.row_iter_mut().enumerate() {
        for (j, e) in row.iter_mut().enumerate() {
            *e = data[i][j*step] as f32;
        }
    }
}

#[inline(always)]
pub fn copy_from_slices_f64(d : &mut DMatrix<f64>, data : &[&[u8]], step : usize) {
    assert!(data.len() > 1);
    let sz = data[0].len();
    assert!(sz / step == d.ncols());
    assert!(data.len() == d.nrows());
    for d in data.iter().skip(1) {
        assert!(d.len() == sz);
    }
    for (i, mut row) in d.row_iter_mut().enumerate() {
        for (j, e) in row.iter_mut().enumerate() {
            *e = data[i][j*step] as f64;
        }
    }
}*/

#[inline(always)]
pub fn subsample<T>(content : &[T], dst : &mut [T], ncols : usize, sample_n : usize)
    where T : Scalar + Copy
{
    assert!(ncols < content.len(), "ncols smaller than content length");
    assert!(content.len() % ncols == 0);
    let nrows = content.len() / ncols;
    let sparse_ncols = if ncols > 1 { ncols / sample_n } else { 1 };
    let sparse_nrows = nrows / sample_n;
    if dst.len() != sparse_nrows * sparse_ncols {
        panic!("Dimension mismatch");
    }
    for r in 0..sparse_nrows {
        for c in 0..sparse_ncols {
            dst[r*sparse_ncols + c] = content[r*sample_n*ncols + c*sample_n];
        }
    }
}

#[inline(always)]
pub fn subsample_convert<T,U>(
    content : &[T],
    dst : &mut [U],
    ncols : usize,
    sample_n : usize,
    transpose : bool
) -> ()
    where
        T : Into<U>,
        T : Scalar + Copy,
        U : Scalar + Copy
{
    assert!(ncols < content.len(), "ncols smaller than content length");
    assert!(content.len() % ncols == 0);
    let nrows = content.len() / ncols;
    let sparse_ncols = if ncols > 1 { ncols / sample_n } else { 1 };
    let sparse_nrows = nrows / sample_n;
    if dst.len() != sparse_nrows * sparse_ncols {
        panic!("Dimension mismatch");
    }
    for r in 0..sparse_nrows {
        for c in 0..sparse_ncols {
            let dst_ix = if transpose { r + c*sparse_nrows } else { r*sparse_ncols + c };
            dst[dst_ix] = content[r*sample_n*ncols + c*sample_n].into();
        }
    }
}

/*#[inline(always)]
pub fn subsample_convert_f32(content : &[u8], dst : &mut [f32], ncols : usize, sample_n : usize,  transpose : bool) {
    assert!(ncols < content.len(), "ncols smaller than content length");
    assert!(content.len() % ncols == 0);
    let nrows = content.len() / ncols;
    let sparse_ncols = if ncols > 1 { ncols / sample_n } else { 1 };
    let sparse_nrows = nrows / sample_n;
    if dst.len() != sparse_nrows * sparse_ncols {
        panic!("Dimension mismatch");
    }
    for r in 0..sparse_nrows {
        for c in 0..sparse_ncols {
            let dst_ix = if transpose { r + c*sparse_nrows } else { r*sparse_ncols + c };
            dst[dst_ix] = content[r*sample_n*ncols + c*sample_n] as f32;
        }
    }
}

#[inline(always)]
pub fn subsample_convert_f64(content : &[u8], dst : &mut [f64], ncols : usize, sample_n : usize,  transpose : bool) {
    assert!(ncols < content.len(), "ncols smaller than content length");
    assert!(content.len() % ncols == 0);
    let nrows = content.len() / ncols;
    let sparse_ncols = if ncols > 1 { ncols / sample_n } else { 1 };
    let sparse_nrows = nrows / sample_n;
    if dst.len() != sparse_nrows * sparse_ncols {
        panic!("Dimension mismatch");
    }
    for r in 0..sparse_nrows {
        for c in 0..sparse_ncols {
            let dst_ix = if transpose { r + c*sparse_nrows } else { r*sparse_ncols + c };
            dst[dst_ix] = content[r*sample_n*ncols + c*sample_n] as f64;
        }
    }
}*/

/* pub fn decode(&self, dec : &[u8]) -> Result<DMatrix<f32>, &'static str> {
        let mut content = Vec::<f32>::new();
        for w in dec.windows(4) {
            let buffer : Result<[u8; 4], _> = w.try_into();
            if let Ok(b) = buffer {
                let u = u32::from_ne_bytes(b);
                content.push(f32::from_bits(u))
            } else {
                return Err("Could not parse buffer as array");
            }
        }
        Ok(DMatrix::from_vec(self.data.nrows(), self.data.ncols(), content))
    }
*/

