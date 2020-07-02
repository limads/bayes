use nalgebra::*;
use simba::scalar::RealField;
use std::convert::TryInto;
use std::sync::{Arc, Mutex};
use nalgebra::storage::*;
//use super::seq::Encoding;
use super::*;
use std::fmt::Debug;

#[cfg(feature="mkl")]
use super::frequency::fft::mkl::FFTPlan;

use crate::signal::FrequencyBasis;
use crate::signal::dwt::gsl::*;

enum Method<N>
    where N : Scalar + From<f32> + Copy + Debug
{
    // N just serves as phantom data if user did not compile with MKL feature.
    None(N),

    #[cfg(feature="mkl")]
    FFT(FFTPlan<N, Dynamic>),

    DWT(DWTPlan<Dynamic>)
}

impl<N> Method<N>
    where N : Scalar + From<f32> + Copy + Debug
{

    #[cfg(feature = "mkl")]
    fn is_fft(&self) -> bool {
        match &self {

            Method::FFT(_) => true,
            _ => false
        }
    }

    fn is_dwt(&self) -> bool {
        match &self {
            Method::DWT(_) => true,
            _ => false
        }
    }

}

/// A structure that converts segments of an two-dimensional
/// 8-bit buffer into a double-precision matrix.
pub struct Surface<N>
    where
        N : Scalar + RealField + Copy + From<f32>
{

    cvt : Option<DMatrix<N>>,

    method : Method<N>,

    // ncols : usize,

    // offset : (usize, usize),

    size : (usize, usize),

}

impl<N> Surface<N>
    where
        N : Scalar + RealField + Copy + From<f32>
{

    pub fn new_empty(nrows : usize, ncols : usize) -> Self {
        let cvt = DMatrix::zeros(nrows, ncols);
        Self{ cvt : Some(cvt), size : (nrows, ncols), method : Method::<N>::None(N::from(0.)) }
    }

    #[cfg(feature = "mkl")]
    pub fn fft(&mut self) -> DMatrixSliceMut<'_, Complex<N>> {
        let cvt = self.cvt.take().unwrap();
        if !self.method.is_fft() {
            self.method = Method::FFT(FFTPlan::new((cvt.nrows(), cvt.ncols())).unwrap());
        }
        if let Method::FFT(ref mut plan) = self.method {
            let ans = plan.forward(&cvt);
            let ans_slice = ans.slice_mut((0,0), (ans.nrows(), ans.ncols()));
            self.cvt = Some(cvt);
            ans_slice
        } else {
            panic!()
        }
    }

    #[cfg(feature = "mkl")]
    pub fn ifft<S>(&self, dst : &mut Matrix<N, Dynamic, Dynamic, S>)
        where
            S : ContiguousStorageMut<N, Dynamic, Dynamic>
    {
        let n = self.cvt.as_ref().unwrap().nrows();
        if !self.method.is_fft() {
            //self.method = Method::FFT(FFTPlan::new((n, 1)).unwrap());
            panic!("Self method should be fft");
        }
        if let Method::FFT(ref plan) = self.method {
            plan.backward_to(dst);
        } else {
            panic!()
        }
    }

    pub fn slice(&self, offset : (usize, usize), size : (usize, usize)) -> DMatrixSlice<'_, N> {
        self.cvt.as_ref().unwrap().slice(offset, size)
    }

    pub fn slice_mut(&mut self, offset : (usize, usize), size : (usize, usize)) -> DMatrixSliceMut<'_, N> {
        self.cvt.as_mut().unwrap().slice_mut(offset, size)
    }

    pub fn copy_from_slices(&mut self, data : &[&[N]]) {
        let cvt = self.cvt.as_mut().unwrap();
        for (i, mut row) in cvt.row_iter_mut().enumerate() {
            for (j, e) in row.iter_mut().enumerate() {
                *e = data[i][j];
            }
        }
    }

    pub fn copy_from_slices_transposed(&mut self, data : &[&[N]]) {
        let cvt = self.cvt.as_mut().unwrap();
        for (mut row, s) in cvt.column_iter_mut().zip(data.iter()) {
            row.copy_from_slice(s);
        }
    }

    /*pub fn update(&'a mut self, source : &'a [u8]) {
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
    }*/

}

impl Surface<f32> {

    pub fn copy_from_raw(&mut self, data : &[&[u8]], enc : Encoding) {
        if enc == Encoding::U8 {
            let cvt = self.cvt.as_mut().unwrap();
            for (i, mut row) in cvt.row_iter_mut().enumerate() {
                for (j, e) in row.iter_mut().enumerate() {
                    *e = data[i][j] as f32;
                }
            }
        } else {
            unimplemented!()
        }
    }

    /// Copy data from a raw memory contiguous u8 buffer.
    pub fn copy_from_raw_strided(&mut self, data : &[u8], stride : usize, enc : Encoding) {
        if enc == Encoding::U8 {
            let cvt = self.cvt.as_mut().unwrap();
            for c in 0..cvt.ncols() {
                convert_f32_slice_strided(
                    data,
                    cvt.column_mut(c).data.as_mut_slice(),
                    stride
                );
            }
        } else {
            unimplemented!()
        }
    }

    pub fn copy_from_raw_transposed(&mut self, data : &[&[u8]], enc : Encoding) {
        if enc == Encoding::U8 {
            let cvt = self.cvt.as_mut().unwrap();
            for (mut col, s) in cvt.column_iter_mut().zip(data.iter()) {
                convert_f32_slice(
                    &s,
                    &mut col.data.as_mut_slice()
                );
            }
        } else {
            unimplemented!()
        }
    }

    /*fn convert(&mut self) {
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
    }*/

}

impl<N> From<(Vec<N>, usize)> for Surface<N>
        where N : Scalar + From<f32> + Copy + Debug + RealField
{

    fn from(data : (Vec<N>, usize)) -> Self {
        let ncols = data.1;
        let nrows = data.0.len() / ncols;
        let dm = DMatrix::from_vec(nrows, ncols, data.0);
        dm.into()
    }

}

impl<N> From<DMatrix<N>> for Surface<N>
    where N : Scalar + From<f32> + Copy + Debug + RealField
{

    fn from(data : DMatrix<N>) -> Self {
        Self {
            size : data.shape(),
            cvt : Some(data),
            method : Method::None(N::from(0.))
        }
    }

}

#[cfg(feature = "mkl")]
#[test]
fn surface_decomposition() {
    let mut surf = Surface::<f64>::new_empty(4,4);
    let row1 : &[u8] = &[1,1,1,1][..];
    let row0 : &[u8] = &[0,0,0,0][..];
    surf.copy_from_raw(&vec![row1, row0, row1, row0][..], Encoding::U8);
    println!("{}", surf.slice((0,0),(4,4)));
    println!("{}", surf.fft());
    for s in surf.dwt() {
        println!("{}", s);
    }
}

impl Surface<f64> {

    pub fn copy_from_raw(&mut self, data : &[&[u8]], enc : Encoding) {
        if enc == Encoding::U8 {
            let cvt = self.cvt.as_mut().unwrap();
            for (i, mut row) in cvt.row_iter_mut().enumerate() {
                for (j, e) in row.iter_mut().enumerate() {
                    *e = data[i][j] as f64;
                }
            }
        } else {
            unimplemented!()
        }
    }

    pub fn copy_from_raw_strided(&mut self, data : &[u8], stride : usize, enc : Encoding) {
        if enc == Encoding::U8 {
            let cvt = self.cvt.as_mut().unwrap();
            for c in 0..cvt.ncols() {
                convert_f64_slice_strided(
                    data,
                    cvt.column_mut(c).data.as_mut_slice(),
                    stride
                );
            }
        } else {
            unimplemented!()
        }
    }

    pub fn copy_from_raw_transposed(&mut self, data : &[&[u8]], enc : Encoding) {
        if enc == Encoding::U8 {
            let cvt = self.cvt.as_mut().unwrap();
            for (mut col, s) in cvt.column_iter_mut().zip(data.iter()) {
                convert_f64_slice(
                    &s,
                    &mut col.data.as_mut_slice()
                );
            }
        } else {
            unimplemented!()
        }
    }

    pub fn dwt(&mut self) -> impl Iterator<Item=DMatrixSlice<'_, f64>> {
        let cvt = self.cvt.take().unwrap();
        if !self.method.is_dwt() {
            self.method = Method::DWT(DWTPlan2D::new(Basis::Daubechies(6, true), cvt.nrows()).unwrap());
        }
        if let Method::DWT(ref mut plan) = self.method {
            let ans = plan.forward(&cvt);
            let levels = plan.iter_levels();
            self.cvt = Some(cvt);
            levels
        } else {
            panic!()
        }
    }

    pub fn idwt<S>(&self, dst : &mut Matrix<f64, Dynamic, Dynamic, S>)
        where
            S : ContiguousStorageMut<f64, Dynamic, Dynamic>
    {
        let n = self.cvt.as_ref().unwrap().nrows();
        if !self.method.is_dwt() {
            //self.method = Method::FFT(FFTPlan::new((n, 1)).unwrap());
            panic!("Self method should be fft");
        }
        if let Method::DWT(ref plan) = self.method {
            plan.backward_to(dst);
        } else {
            panic!()
        }
    }

    /*fn convert(&mut self) {
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
    }*/

}

