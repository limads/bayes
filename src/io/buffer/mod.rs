use nalgebra::*;
// use nalgebra::*;
// use std::ops::{Div, Add, Mul, Sub};
// use std::ops::Range;
// use std::convert::TryFrom;
// use std::io::{self, Write};
use simba::scalar::RealField;

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
pub fn check_bounds(
    start : (usize, usize),
    win_dims : (usize, usize),
    buf_dims : (usize, usize)
) -> bool {
    start.0 + win_dims.0 <= buf_dims.0 && start.1 + win_dims.1 <= buf_dims.1
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

enum Update {
    None,
    Full,
    Partial((usize, usize), (usize, usize))
}

/// Wraps a dynamically-allocated matrix that has its data updated from
/// generic 8 bit buffers such as images (ncols>1) or time streams
/// (ncol == 1). The floating-point data buffer is constant and set
/// at creation. Access to its content is always relative to the
/// last updated position: If the buffer is updated from the full_update,
/// the coordinates of self.slice are the full matrix coordinates; if the buffer is
/// updated from partial_update, the coordinates are relative to the
/// last updated location, and regions outside this update location
/// become unavailable.
pub struct Buffer<N>
    where
        N : Scalar + RealField + From<f32>
{

    data : DMatrix<N>,

    offset : N,

    scale : N,

    last_update : Update
}

impl<N> Buffer<N>
    where N : Scalar + RealField + From<f32>
{

    pub fn create(nrow : usize, ncol : usize) -> Self {
        Self {
            data : DMatrix::zeros(nrow, ncol),
            last_update : Update::None,
            offset : N::from(0.0),
            scale : N::from(1.0)
        }
    }

    pub fn set_offset(&mut self, offset : N) {
        self.offset = offset;
    }

    pub fn set_scale(&mut self, scale : N) {
        self.scale = scale;
    }

    pub fn slice(&self, offset : (usize, usize), size : (usize, usize)) -> DMatrixSlice<'_, N> {
        match self.last_update {
            Update::None => panic!("Buffer was not updated yet"),
            Update::Full => self.data.slice(offset, size),
            Update::Partial(up_off, up_sz) => {
                let off = (up_off.0 + offset.0, up_off.1 + offset.1);
                let sz = (up_sz.0 + size.0, up_sz.1 + size.1);
                self.data.slice(off, sz)
            }
        }
    }

    pub fn slice_mut(&mut self, offset : (usize, usize), size : (usize, usize)) -> DMatrixSliceMut<'_, N> {
        match self.last_update {
            Update::None => panic!("Buffer was not updated yet"),
            Update::Full => self.data.slice_mut(offset, size),
            Update::Partial(up_off, up_sz) => {
                let off = (up_off.0 + offset.0, up_off.1 + offset.1);
                let sz = (up_sz.0 + size.0, up_sz.1 + size.1);
                self.data.slice_mut(off, sz)
            }
        }
    }

}

impl Buffer<f32> {

    pub fn read_full(&mut self, src : &[u8]) {
        self.read_from(src, (0, 0), (self.data.nrows(), self.data.ncols()));
        self.last_update = Update::Full;
    }

    pub fn read_partial(&mut self, src : &[u8], offset : (usize, usize), sz : (usize, usize)) {
        self.read_from(src, offset, sz);
        self.last_update = Update::Partial(offset, sz);
    }

    fn read_from(&mut self, src : &[u8], start : (usize, usize), buf_dims : (usize, usize)) {
        assert!(check_bounds(start, self.data.shape(), buf_dims), "[update_from] Out of buffer bounds");
        let (offset, scale) = (self.offset, self.scale);
        for (buf_row, mut dst_row) in zip_rows_dst_mut(src, &mut self.data, buf_dims.1, start) {
            buf_row.iter().zip(dst_row.iter_mut())
                .for_each(|(b, d)|{ *d = (*b as f32+offset)*scale; })
        }
    }

    pub fn write_into(&self, dst : &mut [u8], start : (usize, usize), buf_dims : (usize, usize)) {
        assert!(check_bounds(start, self.data.shape(), buf_dims), "[write_into] Out of buffer bounds");
        let (offset, scale) = (self.offset, self.scale);
        for (buf_row, dst_row) in zip_rows_buffer_mut(dst, &self.data, buf_dims.1, start) {
        buf_row.iter_mut().zip(dst_row.iter())
            .for_each(|(b, d)|{ *b = ((*d + offset)*scale) as u8 })
        }
    }

}

impl Buffer<f64> {

    pub fn full_update(&mut self, _src : &[u8]) {

    }


    pub fn partial_update(&mut self, _src : &[u8]) {

    }

}

