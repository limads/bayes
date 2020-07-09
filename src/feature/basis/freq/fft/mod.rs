use nalgebra::*;
use super::*;
use nalgebra::storage::*;
use simba::scalar::RealField;
use std::fmt::Debug;
use crate::feature::basis::*;

/// Wrappers over MKL FFT routines.
pub mod mkl;

use mkl::*;

pub struct FFT<N>
    where
        N : Scalar + Debug + Copy + From<f32> + RealField,
        Complex<N> : Scalar + Clone
{
    plan : FFTPlan<N, U1>,
    domain : Option<DVector<N>>,
    back_call : bool
}

impl FFT<f32> {

    pub fn new<S : Into<DVector<f32>>>(s : S) -> Self {
        let domain : DVector<f32> = s.into();
        let mut plan = FFTPlan::new((domain.nrows(), 1)).unwrap();
        plan.forward(&domain);
        let fft = Self{ plan, domain : Some(domain), back_call : true };
        fft
    }

    pub fn new_empty(nrow : usize) -> Self {
        Self::new(DVector::from_element(nrow, 0.0))
    }
}

impl FFT<f64> {

    pub fn new<S : Into<DVector<f64>>>(s : S) -> Self {
        let domain : DVector<f64> = s.into();
        let mut plan = FFTPlan::new((domain.nrows(), 1)).unwrap();
        plan.forward(&domain);
        let fft = Self{ plan, domain : Some(domain), back_call : true };
        fft
    }

    pub fn new_empty(nrows : usize) -> Self {
        Self::new(DVector::from_element(nrows, 0.0))
    }
}

impl<'a, N> Basis<'a, N, Complex<N>, U1> for FFT<N>
    where
        N : Scalar + Debug + Copy + From<f32> + RealField,
        Complex<N> : Scalar + Clone
{

    fn forward<S>(&'a mut self, s : &Matrix<N, Dynamic, U1, S>) -> &'a DVector<Complex<N>>
        where S : ContiguousStorage<N, Dynamic, U1>
    {
        //let s : DVectorSlice<'a, N> = s.into();
        let ans = self.plan.forward(&s);
        self.back_call = false;
        &(*ans)
    }

    fn backward(&'a mut self) ->  &'a DVector<N> {
        let mut b_buf = self.domain.take().unwrap();
        self.plan.backward_to(&mut b_buf);
        self.domain = Some(b_buf);
        self.back_call = true;
        self.domain.as_ref().unwrap()
    }

    fn partial_backward<S>(&'a mut self, n : usize) -> DVectorSlice<'a, N> {
        //let mut dst = self.plan.take().unwrap();
        //plan.backward_to(&mut dst);
        //self.dst = Some(dst);
        //dst.as_ref().into()
        unimplemented!()
    }

    fn coefficients(&'a self) ->  &'a DVector<Complex<N>> {
        &self.plan.forward_buffer
    }

    fn coefficients_mut(&'a mut self) ->  &'a mut DVector<Complex<N>> {
        &mut self.plan.forward_buffer
    }

    fn domain(&'a self) -> Option<&'a DVector<N>> {
        if self.back_call {
            self.domain.as_ref()
        } else {
            None
        }
    }

    fn domain_mut(&'a mut self) -> Option<&'a mut DVector<N>> {
        if self.back_call {
            self.domain.as_mut()
        } else {
            None
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
impl<'a, N> Basis<'a, N, Complex<N>, Dynamic> for FFT2D<N>
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



