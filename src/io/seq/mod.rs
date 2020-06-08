use nalgebra::*;
use simba::scalar::RealField;
use std::convert::TryInto;
use std::sync::{Arc, Mutex};
use nalgebra::storage::*;

/// A referential structure that converts segments of an one-dimensional
/// 8-bit buffer into a double-precision vector.
pub struct Sequence<'a, N>
    where
        N : Scalar + RealField
{

    source : &'a [u8],

    offset : usize,

    size : usize,

    cvt : DVector<N>,

}

impl<'a, N> Sequence<'a, N>
    where
        N : Scalar + RealField
{

    pub fn new(source : &'a [u8]) -> Self {
        let size = source.len();
        let cvt = DVector::zeros(size);
        Self{ source, cvt, offset : 0, size }
    }

    pub fn update(&mut self, source : &'a [u8]) {
        self.source = source;
    }

    pub fn reposition(&mut self, offset : usize, size : usize) {
        self.offset = offset;
        self.size = size;
    }

}

impl<'a> Sequence<'a, f32> {

    pub fn extract(&mut self) -> DVectorSlice<'_, f32> {
        convert_f32_slice(
            &self.source[self.offset..(self.offset+self.size)],
            &mut self.cvt.data.as_mut_slice()[self.offset..(self.offset+self.size)]
        );
        self.cvt.rows(self.offset, self.size)
    }

    pub fn extract_mut(&mut self) -> DVectorSliceMut<'_, f32> {
        convert_f32_slice(
            &self.source[self.offset..(self.offset+self.size)],
            &mut self.cvt.data.as_mut_slice()[self.offset..(self.offset+self.size)]
        );
        self.cvt.rows_mut(self.offset, self.size)
    }

    pub fn view(&self) -> DVectorSlice<'_,f32> {
        self.cvt.rows(self.offset, self.size)
    }

}

impl<'a> Sequence<'a, f64> {

    pub fn extract(&mut self) -> DVectorSlice<'_, f64> {
        convert_f64_slice(
            &self.source[self.offset..(self.offset+self.size)],
            &mut self.cvt.data.as_mut_slice()[self.offset..(self.offset+self.size)]
        );
        self.cvt.rows(self.offset, self.size)
    }

    pub fn extract_mut(&mut self) -> DVectorSliceMut<'_, f64> {
        convert_f64_slice(
            &self.source[self.offset..(self.offset+self.size)],
            &mut self.cvt.data.as_mut_slice()[self.offset..(self.offset+self.size)]
        );
        self.cvt.rows_mut(self.offset, self.size)
    }

    pub fn view(&self) -> DVectorSlice<'_,f64> {
        self.cvt.rows(self.offset, self.size)
    }

}

#[inline(always)]
fn convert_f32_slice(src : &[u8], dst : &mut [f32]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as f32;
    }
}

#[inline(always)]
fn convert_f64_slice(src : &[u8], dst : &mut [f64]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as f64;
    }
}

/*
pub fn decode(&self, dec : &[u8]) -> Result<DMatrix<f32>, &'static str> {
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


