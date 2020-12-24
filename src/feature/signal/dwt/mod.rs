use nalgebra::*;
use iter::*;
use nalgebra::storage::*;
use gsl::*;
use crate::feature::signal::*;

#[derive(Clone)]
pub enum Basis {
    // Centered or not
    Haar(bool),

    // k = 4..20 even; centered or not
    Daubechies(usize, bool),

    // k = 103, 105, 202..208 even, 301..309 odd; centered or not
    BSpline(usize, bool)
}

// Utilities for iterating over the levels of a wavelet transform.
pub mod iter;

/// Wrappers over GSL wavelet routines.
pub (crate) mod gsl;

/// Wavelet transform.
pub struct Wavelet {
    plan : DWTPlan
}

impl Wavelet {

    pub fn new(basis : Basis, sz : usize) -> Result<Self, &'static str> {
        Ok(Self { plan : DWTPlan::new(basis, (sz, 1))? })
    }
    
    pub fn forward_mut(&self, src : &Signal<f64>, dst : &mut Pyramid<f64>) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    pub fn forward(&self, src : &Signal<f64>) -> Pyramid<f64> {
        let mut dst = Pyramid::new_constant(self.plan.shape().0, 0.0);
        self.forward_mut(src, &mut dst);
        dst
    }
    
    pub fn backward_mut(&self, src : &Pyramid<f64>, dst : &mut Signal<f64>) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    pub fn backward(&self, src : &Pyramid<f64>) -> Signal<f64> {
        let mut dst = Signal::new_constant(self.plan.shape().0, 0.0);
        self.backward_mut(src, &mut dst);
        dst
    }
}

/*pub struct Scale<'a> {
    vals : DVectorSliceMut<'a, f64>
}

impl Iterator<Item=Scale<'a>> for ScaleIter {

    fn next(&mut self) -> Option<Scale<'a>> {
        if self.curr_lvl == self.max_lvl + 1 {
            return None;
        }
        self.curr_lvl += 1;
        Some(())
    }
}*/
/*struct ScaleMut<'a, N> {
    full : &'a mut Pyramid<N>
    curr : usize
}*/

/// Output of a wavelet transform.
#[derive(Clone, Debug)]
pub struct Pyramid<N> 
where
    N : Scalar
{
    buf : DVector<N>
}

impl Pyramid<f64> {

    pub fn new_constant(n : usize, value : f64) -> Self {
        Self{ buf : DVector::from_element(n, value) }
    }
    
    pub fn levels<'a>(&'a self) -> impl Iterator<Item=DVectorSlice<'a, f64>> {
        DWTIteratorBase::<&'a DVector<f64>>::new_ref(&self.buf)
    }
    
    pub fn levels_mut<'a>(&'a mut self) -> impl Iterator<Item=DVectorSliceMut<'a, f64>> {
        DWTIteratorBase::<&'a mut DVector<f64>>::new_mut(&mut self.buf)
    }
}

impl<N> AsRef<[N]> for Pyramid<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &[N] {
        self.buf.data.as_slice()
    }
}

impl<N> AsMut<[N]> for Pyramid<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut [N] {
        self.buf.data.as_mut_slice()
    }
}

impl<N> AsRef<DVector<N>> for Pyramid<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DVector<N> {
        &self.buf
    }
}

impl<N> From<DVector<N>> for Pyramid<N> 
where
    N : Scalar
{
    fn from(s : DVector<N>) -> Self {
        Self{ buf : s }
    }
}

impl<N> From<Vec<N>> for Pyramid<N> 
where
    N : Scalar
{
    fn from(s : Vec<N>) -> Self {
        Self{ buf : DVector::from_vec(s) }
    }
}

/*impl Forward<Signal<f64>> for Wavelet {
    
    type Output = Signal<f64>;
    
    fn forward_mut(&self, src : &Signal<f64>, dst : &mut Self::Output) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    fn forward(&self, src : &Signal<f64>) -> Self::Output {
        let mut dst = Signal::new_constant(self.plan.shape().0, 0.0);
        self.forward_mut(src, &mut dst);
        dst
    }
}

impl Backward<Signal<f64>> for Wavelet {
    
    type Output = Signal<f64>;
    
    fn backward_mut(&self, src : &Signal<f64>, dst : &mut Self::Output) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    fn backward(&self, src : &Signal<f64>) -> Self::Output {
        let mut dst = Signal::new_constant(self.plan.shape().0, 0.0);
        self.backward_mut(src, &mut dst);
        dst
    }
    
}*/

/*#[derive(Clone)]
pub struct DWT {
    plan : DWTPlan1D,
    domain : Option<DVector<f64>>,
    // back_call : bool
}

impl DWT {

    /// Builds a new 1D discrete wavelet transform from the zeroed-out
    /// buffer with the given number of rows.
    pub fn new_empty(nrows : usize) -> Self {
        Self::new(DVector::zeros(nrows))
    }

    /// Builds a new 1D discrete wavelet transform from the informed source.
    pub fn new<S>(s : S) -> Self
        where S : Into<DVector<f64>>
    {
        let mut vs = s.into();
        let mut plan = DWTPlan1D::new(gsl::Basis::Daubechies(6, true), vs.nrows()).unwrap();
        let _ = plan.forward(&vs);
        let domain = Some(vs);
        //let back_call = true;
        //Self{ plan, domain, back_call }
        Self{ plan, domain }
    }

    pub fn iter_levels<'a>(&'a self) -> DWTIterator1D<'a> {
        self.plan.iter_levels()
    }

    pub fn iter_levels_mut<'a>(&'a mut self) -> DWTIteratorMut1D<'a> {
        self.plan.iter_levels_mut()
    }

}

impl<'a, S> Forward<'a, Matrix<f64, Dynamic, U1, S>, DVector<f64>> for DWT
where S : ContiguousStorage<f64, Dynamic, U1>
{

    fn forward_from(&'a mut self, s : &Matrix<f64, Dynamic, U1, S>) -> &'a DVector<f64> {
        let _ = self.plan.forward(&s);
        &self.plan.buf
        //self.back_call = false;
        //&(*ans)
    }

    /*fn partial_backward<S>(&'a mut self, n : usize) -> DVectorSlice<'a, f64> {
        //let mut dst = self.plan.take().unwrap();
        //plan.backward_to(&mut dst);
        //self.dst = Some(dst);
        //dst.as_ref().into()
        unimplemented!()
    }*/

    fn coefficients(&'a self) ->  &'a DVector<f64> {
        &self.plan.buf
    }

    fn coefficients_mut(&'a mut self) ->  &'a mut DVector<f64> {
        &mut self.plan.buf
    }

}

impl<'a, /*S*/ > Backward<'a, DVector<f64>, DVector<f64>> for DWT
where
    // S : ContiguousStorageMut<f64, Dynamic, U1>
{

    fn backward_from(&'a mut self, coefs : &'a DVector<f64>) -> Option<&'a DVector<f64>> {
        self.plan.buf.copy_from(&coefs);
        self.backward_from_self()
    }

    fn backward_from_self(&'a mut self) -> Option<&'a DVector<f64>> {
        if let Some(mut dom) = self.take_domain() {
            self.backward_mut(&mut dom);
            self.domain = Some(dom);
            self.domain()
        } else {
            None
        }
    }

    fn backward_mut(&self, dst : &mut DVector<f64>) {
        // let mut b_buf = self.domain.take().unwrap();
        self.plan.backward_to(dst);
        // self.domain = Some(b_buf);
        // self.back_call = true;
        // self.domain.as_ref().unwrap()
    }

    fn take_domain(&mut self) -> Option<DVector<f64>> {
        self.domain.take()
    }

    fn domain(&'a self) -> Option<&'a DVector<f64>> {
        self.domain.as_ref()
    }

    fn domain_mut(&'a mut self) -> Option<&'a mut DVector<f64>> {
        self.domain.as_mut()
    }
}

pub struct DWT2D {
    plan : DWTPlan2D,
    domain : Option<DMatrix<f64>>,
    // back_call : bool
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
        //let back_call = true;
        //Self{ plan, domain, back_call }
        Self{ plan, domain }
    }

    pub fn iter_levels<'a>(&'a self) -> DWTIterator2D<'a> {
        self.plan.iter_levels()
    }

    pub fn iter_levels_mut<'a>(&'a mut self) -> DWTIteratorMut2D<'a> {
        self.plan.iter_levels_mut()
    }

}

impl<'a, S> Forward<'a, Matrix<f64, Dynamic, Dynamic, S>, DMatrix<f64>> for DWT2D
where S : ContiguousStorage<f64, Dynamic, Dynamic>
{

    fn forward_from(&'a mut self, s : &Matrix<f64, Dynamic, Dynamic, S>) -> &'a DMatrix<f64> {
        let _ = self.plan.forward(&s);
        &self.plan.buf
        // self.back_call = false;
        // &(*ans)
    }

    // fn partial_backward<S>(&'a mut self, n : usize) -> DMatrixSlice<'a, f64> {
    //    unimplemented!()
    // }

    fn coefficients(&'a self) -> &'a DMatrix<f64> {
        &self.plan.buf
    }

    fn coefficients_mut(&'a mut self) -> &'a mut DMatrix<f64> {
        &mut self.plan.buf
    }

}

impl<'a, /*S*/ > Backward<'a, DMatrix<f64>, DMatrix<f64>> for DWT2D
where
    //S : ContiguousStorageMut<f64, Dynamic, Dynamic>
{

    fn backward_from(&'a mut self, coefs : &'a DMatrix<f64>) -> Option<&'a DMatrix<f64>> {
        self.plan.buf.copy_from(&coefs);
        self.backward_from_self()
    }

    fn backward_from_self(&'a mut self) -> Option<&'a DMatrix<f64>> {
        if let Some(mut dom) = self.take_domain() {
            self.backward_mut(&mut dom);
            self.domain = Some(dom);
            self.domain()
        } else {
            None
        }
    }

    fn backward_mut(&self, dst : &mut DMatrix<f64>) {
        // let mut b_buf = self.domain.take().unwrap();
        self.plan.backward_to(dst);
        // self.domain = Some(b_buf);
        // self.back_call = true;
        // self.domain.as_ref().unwrap()
    }

    fn take_domain(&mut self) -> Option<DMatrix<f64>> {
        self.domain.take()
    }

    fn domain(&'a self) -> Option<&'a DMatrix<f64>> {
        self.domain.as_ref()
    }

    fn domain_mut(&'a mut self) -> Option<&'a mut DMatrix<f64>> {
        self.domain.as_mut()
    }
}*/


