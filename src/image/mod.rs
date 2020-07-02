use nalgebra::*;
use super::*;
use nalgebra::storage::*;
use simba::scalar::RealField;
use std::fmt::Debug;
use super::signal::{self, *};
use super::signal::dwt::{*, gsl::*, iter::*};
use super::signal::sampling::*;

#[cfg(feature = "mkl")]
use super::signal::fft::mkl::*;

impl Signal<u8, f64> for DMatrix<f64> {

    fn resample(&mut self, src : &[u8], step : usize) {
        if step == 1 {
            convert_f64_slice(
                &src,
                self.data.as_mut_slice()
            );
        } else {
            let ncols = self.ncols();
            assert!(src.len() / step == self.nrows(), "Dimension mismatch");
            subsample_convert_f64(&src, self.data.as_mut_slice(), ncols, step, true);
        }
    }

}

impl Signal<u8, f32> for DMatrix<f32> {

    fn resample(&mut self, src : &[u8], step : usize) {
        if step == 1 {
            convert_f32_slice(
                &src,
                self.data.as_mut_slice()
            );
        } else {
            let ncols = self.ncols();
            assert!(src.len() / step == self.nrows(), "Dimension mismatch");
            subsample_convert_f32(&src, self.data.as_mut_slice(), ncols, step, true);
        }
    }

}

#[cfg(feature = "mkl")]
pub struct FFT2D<N>
    where
        N : Scalar + Debug + Copy + From<f32> + RealField,
        Complex<N> : Scalar + Clone
{
    plan : FFTPlan<N, Dynamic>,
    domain : Option<DMatrix<N>>,
    back_call : bool
}

#[cfg(feature = "mkl")]
impl<'a, N> signal::Transform<'a, N, Complex<N>, Dynamic> for FFT2D<N>
    where
        N : Scalar + Debug + Copy + From<f32> + RealField,
        Complex<N> : Scalar + Clone
{

    fn forward<S>(&'a mut self, s : &Matrix<N, Dynamic, Dynamic, S>) -> &'a DMatrix<Complex<N>>
        where S : ContiguousStorage<N, Dynamic, Dynamic>
    {
        //let s : Matrix<N, Dynamic, Dynamic, SliceStorage<'a, N, Dynamic, Dynamic, U1, Dynamic>> = s.into();
        let ans = self.plan.forward(&s);
        self.back_call = false;
        &(*ans)
    }

    fn backward(&'a mut self) -> &'a DMatrix<N> {
        let mut b_buf = self.domain.take().unwrap();
        self.plan.backward_to(&mut b_buf);
        self.domain = Some(b_buf);
        self.back_call = true;
        self.domain.as_ref().unwrap()
    }

    fn partial_backward<S>(&'a mut self, n : usize) -> DMatrixSlice<'a, N> {
        unimplemented!()
    }

    fn coefficients(&'a self) -> &'a DMatrix<Complex<N>> {
        &self.plan.forward_buffer
    }

    fn coefficients_mut(&'a mut self) -> &'a mut DMatrix<Complex<N>> {
        &mut self.plan.forward_buffer
    }

    fn domain(&'a self) -> Option<&'a DMatrix<N>> {
        if self.back_call {
            self.domain.as_ref()
        } else {
            None
        }
    }

    fn domain_mut(&'a mut self) -> Option<&'a mut DMatrix<N>> {
        if self.back_call {
            self.domain.as_mut()
        } else {
            None
        }
    }

}

#[cfg(feature = "mkl")]
impl FFT2D<f32> {

    pub fn new<S : Into<DMatrix<f32>>>(s : S) -> Self {
        let domain : DMatrix<f32> = s.into();
        let mut plan = FFTPlan::new((domain.nrows(), domain.ncols())).unwrap();
        plan.forward(&domain);
        let fft = Self{ plan, domain : Some(domain), back_call : true };
        fft
    }

    pub fn new_empty(nrow : usize, ncol : usize) -> Self {
        Self::new(DMatrix::from_element(nrow, ncol, 0.0))
    }
}

#[cfg(feature = "mkl")]
impl FFT2D<f64> {

    pub fn new<S : Into<DMatrix<f64>>>(s : S) -> Self {
        let domain : DMatrix<f64> = s.into();
        let mut plan = FFTPlan::new((domain.nrows(), domain.ncols())).unwrap();
        plan.forward(&domain);
        let fft = Self{ plan, domain : Some(domain), back_call : true };
        fft
    }

    pub fn new_empty(nrow : usize, ncol : usize) -> Self {
        Self::new(DMatrix::from_element(nrow, ncol, 0.0))
    }
}

pub struct DWT2D {
    plan : DWTPlan2D,
    domain : Option<DMatrix<f64>>,
    back_call : bool
}

impl DWT2D {

    pub fn new_empty(nrows : usize, ncols : usize) -> Self {
        Self::new(DMatrix::zeros(nrows, ncols))
    }

    pub fn new<S>(s : S) -> Self
        where S : Into<DMatrix<f64>>
    {
        let mut ms : DMatrix<f64> = s.into();
        let mut plan = DWTPlan2D::new(Basis::Daubechies(6, true), ms.nrows()).unwrap();
        let _ = plan.forward(&ms);
        let domain = Some(ms);
        let back_call = true;
        Self{ plan, domain, back_call }
    }

    pub fn iter_levels<'a>(&'a self) -> DWTIterator2D<'a> {
        self.plan.iter_levels()
    }

    pub fn iter_levels_mut<'a>(&'a mut self) -> DWTIteratorMut2D<'a> {
        self.plan.iter_levels_mut()
    }

}

impl<'a> signal::Transform<'a, f64, f64, Dynamic> for DWT2D {

    fn forward<S>(&'a mut self, s : &Matrix<f64, Dynamic, Dynamic, S>) -> &'a DMatrix<f64>
        where S : ContiguousStorage<f64, Dynamic, Dynamic>
    {
        let ans = self.plan.forward(&s);
        self.back_call = false;
        &(*ans)
    }

    fn backward(&'a mut self) -> &'a DMatrix<f64> {
        let mut b_buf = self.domain.take().unwrap();
        self.plan.backward_to(&mut b_buf);
        self.domain = Some(b_buf);
        self.back_call = true;
        self.domain.as_ref().unwrap()
    }

    fn partial_backward<S>(&'a mut self, n : usize) -> DMatrixSlice<'a, f64> {
        unimplemented!()
    }

    fn coefficients(&'a self) -> &'a DMatrix<f64> {
        &self.plan.buf
    }

    fn coefficients_mut(&'a mut self) -> &'a mut DMatrix<f64> {
        &mut self.plan.buf
    }

    fn domain(&'a self) -> Option<&'a DMatrix<f64>> {
        if self.back_call {
            self.domain.as_ref()
        } else {
            None
        }
    }

    fn domain_mut(&'a mut self) -> Option<&'a mut DMatrix<f64>> {
        if self.back_call {
            self.domain.as_mut()
        } else {
            None
        }
    }

}


