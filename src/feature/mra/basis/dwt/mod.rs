use nalgebra::*;
use iter::*;
use nalgebra::storage::*;
use gsl::*;
use super::{Basis, FrequencyBasis};

/// Utilities for iterating over the levels of a wavelet transform.
pub mod iter;

/// Wrappers over GSL wavelet routines.
pub mod gsl;

pub struct DWT {
    plan : DWTPlan1D,
    domain : Option<DVector<f64>>,
    back_call : bool
}

impl DWT {

    pub fn new_empty(nrows : usize) -> Self {
        Self::new(DVector::zeros(nrows))
    }

    pub fn new<S>(s : S) -> Self
        where S : Into<DVector<f64>>
    {
        let mut vs = s.into();
        let mut plan = DWTPlan1D::new(gsl::Basis::Daubechies(6, true), vs.nrows()).unwrap();
        let _ = plan.forward(&vs);
        let domain = Some(vs);
        let back_call = true;
        Self{ plan, domain, back_call }
    }

    pub fn iter_levels<'a>(&'a self) -> DWTIterator1D<'a> {
        self.plan.iter_levels()
    }

    pub fn iter_levels_mut<'a>(&'a mut self) -> DWTIteratorMut1D<'a> {
        self.plan.iter_levels_mut()
    }

}

impl<'a> Basis<'a, f64, f64, U1> for DWT {

    fn forward<S>(&'a mut self, s : &Matrix<f64, Dynamic, U1, S>) -> &'a DVector<f64>
        where S : ContiguousStorage<f64, Dynamic, U1>
    {
        let ans = self.plan.forward(&s);
        self.back_call = false;
        &(*ans)
    }

    fn backward(&'a mut self) ->  &'a DVector<f64> {
        let mut b_buf = self.domain.take().unwrap();
        self.plan.backward_to(&mut b_buf);
        self.domain = Some(b_buf);
        self.back_call = true;
        self.domain.as_ref().unwrap()
    }

    fn partial_backward<S>(&'a mut self, n : usize) -> DVectorSlice<'a, f64> {
        //let mut dst = self.plan.take().unwrap();
        //plan.backward_to(&mut dst);
        //self.dst = Some(dst);
        //dst.as_ref().into()
        unimplemented!()
    }

    fn coefficients(&'a self) ->  &'a DVector<f64> {
        &self.plan.buf
    }

    fn coefficients_mut(&'a mut self) ->  &'a mut DVector<f64> {
        &mut self.plan.buf
    }

    fn domain(&'a self) -> Option<&'a DVector<f64>> {
        if self.back_call {
            self.domain.as_ref()
        } else {
            None
        }
    }

    fn domain_mut(&'a mut self) -> Option<&'a mut DVector<f64>> {
        if self.back_call {
            self.domain.as_mut()
        } else {
            None
        }
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
        let mut plan = DWTPlan2D::new(gsl::Basis::Daubechies(6, true), ms.nrows()).unwrap();
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

impl<'a> Basis<'a, f64, f64, Dynamic> for DWT2D {

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

