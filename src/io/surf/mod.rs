use nalgebra::*;
use simba::scalar::RealField;
use std::convert::TryInto;
use std::sync::{Arc, Mutex};
use nalgebra::storage::*;

/// A referential structure that converts segments of an two-dimensional
/// 8-bit buffer into a double-precision matrix.
pub struct Surface<'a, N>
    where
        N : Scalar + RealField
{

    source : &'a [u8],

    cvt : DMatrix<N>,

    ncols : usize,

    offset : (usize, usize),

    size : (usize, usize),

}

impl<'a, N> Surface<'a, N>
    where
        N : Scalar + RealField
{

    pub fn new(source : &'a [u8], ncols : usize) -> Self {
        let win = source.len();
        let nrows = source.len() / ncols;
        let cvt = DMatrix::zeros(nrows, ncols);
        Self{ source, cvt, offset : (0, 0), size : (nrows, ncols), ncols }
    }

    pub fn update(&'a mut self, source : &'a [u8]) {
        self.source = source;
    }

    pub fn reposition(&mut self, offset : (usize, usize), size : (usize, usize)) {
        self.offset = offset;
        self.size = size;
    }

    fn slice_extensions(&self, c : usize) -> (usize, usize) {
        let start = self.offset.0 * self.ncols + self.offset.1 + c;
        let end = start + self.ncols*(self.size.0-1) + 1;
        (start, end)
    }

}

impl<'a> Surface<'a, f32> {

    fn convert(&mut self) {
        for c in 0..self.size.1 {
            let (start, end) = self.slice_extensions(c);
            convert_f32_slice_strided(
                &self.source[start..end],
                self.cvt.slice_mut(self.offset, self.size).column_mut(c).data.as_mut_slice(),
                self.ncols
            );
        }
    }

    pub fn extract_mut(&mut self) -> DMatrixSliceMut<'_, f32> {
        self.convert();
        self.cvt.slice_mut(self.offset, self.size)
    }

    pub fn extract(&mut self) -> DMatrixSlice<'_, f32> {
        self.convert();
        self.cvt.slice(self.offset, self.size)
    }

    pub fn view(&self) -> DMatrixSlice<'_, f32> {
        self.cvt.slice(self.offset, self.size)
    }

}

impl<'a> Surface<'a, f64> {

    fn convert(&mut self) {
        for c in 0..self.size.1 {
            let (start, end) = self.slice_extensions(c);
            convert_f64_slice_strided(
                &self.source[start..end],
                self.cvt.slice_mut(self.offset, self.size).column_mut(c).data.as_mut_slice(),
                self.ncols
            );
        }
    }

    pub fn extract_mut(&mut self) -> DMatrixSliceMut<'_, f64> {
        self.convert();
        self.cvt.slice_mut(self.offset, self.size)
    }

    pub fn extract(&mut self) -> DMatrixSlice<'_, f64> {
        self.convert();
        self.cvt.slice(self.offset, self.size)
    }

    pub fn view(&self) -> DMatrixSlice<'_, f64> {
        self.cvt.slice(self.offset, self.size)
    }

}

#[inline(always)]
fn convert_f32_slice_strided(src : &[u8], dst : &mut [f32], cstride : usize) {
    for i in 0..dst.len() {
        dst[i] = src[i*cstride] as f32
    }
}

#[inline(always)]
fn convert_f64_slice_strided(src : &[u8], dst : &mut [f64], cstride : usize) {
    for i in 0..dst.len() {
        dst[i] = src[i*cstride] as f64
    }
}


