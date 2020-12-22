use nalgebra::*;
use super::*;
use nalgebra::storage::*;
use simba::scalar::RealField;
use std::fmt::Debug;
use super::*;
use super::conv::*;
use num_traits::Num;

/// Wrappers over MKL FFT routines.
pub(crate) mod mkl;

use mkl::*;

/*pub enum Overlap {
    None,
    Half,
    Quarter
}*/

/// Fourier transform
pub struct Fourier<N> 
where
    N : Scalar,
    Complex<N> : Scalar
{
    plan : FFTPlan<N>
}

impl Fourier<f32> {

    pub fn new(sz : usize) -> Result<Self, String> {
        Ok(Self { plan : FFTPlan::<f32>::new((sz, 1))? })
    }
}

impl Fourier<f64> {

    pub fn new(sz : (usize, usize)) -> Result<Self, String> {
        Ok(Self { plan : FFTPlan::<f64>::new(sz)? })
    }
    
}

/// Output of a fourier transform.
pub struct Spectrum<N> 
where
    N : Scalar + Copy
{
    buf : DVector<Complex<N>>
}

impl<N> Spectrum<N> 
where
    N : Scalar + Copy
{
    pub fn new_constant(n : usize, value : Complex<N>) -> Self {
        Self{ buf : DVector::from_element(n, value) }
    }
}

impl<N> AsRef<[Complex<N>]> for Spectrum<N> 
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &[Complex<N>] {
        self.buf.data.as_slice()
    }
}

impl<N> AsMut<[Complex<N>]> for Spectrum<N> 
where
    N : Scalar + Copy
{
    fn as_mut(&mut self) -> &mut [Complex<N>] {
        self.buf.data.as_mut_slice()
    }
}

impl<N> AsRef<DVector<Complex<N>>> for Spectrum<N> 
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &DVector<Complex<N>> {
        &self.buf
    }
}

impl<N> From<DVector<Complex<N>>> for Spectrum<N> 
where
    N : Scalar + Copy
{
    fn from(s : DVector<Complex<N>>) -> Self {
        Self{ buf : s }
    }
}

impl<N> From<Vec<Complex<N>>> for Spectrum<N> 
where
    N : Scalar + Copy
{
    fn from(s : Vec<Complex<N>>) -> Self {
        Self{ buf : DVector::from_vec(s) }
    }
}

/// Slice over a spectrum.
pub struct Band {

}

impl<N> Fourier<N>
where
    N : Scalar + From<f32> + Num + Copy,
    Complex<N> : Scalar
{
    pub fn forward_mut(&self, src : &Signal<N>, dst : &mut Spectrum<N>) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    pub fn forward(&self, src : &Signal<N>) -> Spectrum<N> {
        let zero = N::from(0.0 as f32);
        let mut dst = Spectrum::new_constant(self.plan.shape().0, Complex::new(zero.clone(), zero));
        self.forward_mut(src, &mut dst);
        dst
    }
    
    pub fn backward_mut(&self, src : &Spectrum<N>, dst : &mut Signal<N>) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    pub fn backward(&self, src : &Spectrum<N>) -> Signal<N> {
        let mut dst = Signal::new_constant(self.plan.shape().0, N::from(0.0 as f32));
        self.backward_mut(src, &mut dst);
        dst
    }
}

/*impl<N> Forward<Signal<N>> for Fourier<N> 
where
    N : Scalar + From<f32> + Num,
    Complex<N> : Scalar
{
    
    type Output = Signal<Complex<N>>;
    
    fn forward_mut(&self, src : &Signal<N>, dst : &mut Self::Output) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    fn forward(&self, src : &Signal<N>) -> Self::Output {
        let zero = N::from(0.0 as f32);
        let mut dst = Signal::new_constant(self.plan.shape().0, Complex::new(zero.clone(), zero));
        self.forward_mut(src, &mut dst);
        dst
    }
}

impl<N> Backward<Signal<Complex<N>>> for Fourier<N> 
where
    N : Scalar + From<f32>,
    Complex<N> : Scalar
{
    
    type Output = Signal<N>;
    
    fn backward_mut(&self, src : &Signal<Complex<N>>, dst : &mut Self::Output) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    fn backward(&self, src : &Signal<Complex<N>>) -> Self::Output {
        let mut dst = Signal::new_constant(self.plan.shape().0, N::from(0.0 as f32));
        self.backward_mut(src, &mut dst);
        dst
    }
}*/

fn half_circular_domain<N : Scalar + From<f32>>(sz : usize) -> DVector<N> {
    let samples : Vec<N> = (0..sz).map(|s| {
        N::from((s as f32 * std::f32::consts::PI) / sz as f32)
    }).collect();
    DVector::from_vec(samples)
}

fn hann<N : Scalar + Into<f64> + From<f32>>(sz : usize) -> DVector<N> {
    let samples = half_circular_domain::<f32>(sz);
    DVector::<N>::from_iterator(sz,
        samples.iter().map(|x| N::from((*x as f32).sin().powf(2.0)))
    )
}

fn hann2<N : Scalar + Into<f64> + From<f32>>(sz : usize) -> DMatrix<N> {
    let samples = hann::<f32>(sz);
    let mut ans = DMatrix::<N>::from_element(sz, sz, N::from(0.0));
    for (i, r) in samples.iter().enumerate() {
        for (j, c) in samples.iter().enumerate() {
            ans[(i,j)] = N::from((r * c).sqrt());
        }
    }
    ans
}

/*pub struct FFT<N>
    where
        N : Scalar + Debug + Copy + From<f32> + RealField,
        Complex<N> : Scalar + Clone
{
    plan : FFTPlan<N, U1>,
    domain : Option<DVector<N>>,
    // back_call : bool
}

impl FFT<f32> {

    pub fn new<S : Into<DVector<f32>>>(s : S) -> Self {
        let domain : DVector<f32> = s.into();
        let mut plan = FFTPlan::new((domain.nrows(), 1)).unwrap();
        plan.forward(&domain);
        // let fft = Self{ plan, domain : Some(domain), back_call : true };
        // fft
        Self { plan, domain : Some(domain) }
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
        // let fft = Self{ plan, domain : Some(domain), back_call : true };
        // fft
        Self{ plan, domain : Some(domain) }
    }

    pub fn new_empty(nrows : usize) -> Self {
        Self::new(DVector::from_element(nrows, 0.0))
    }
}

impl<'a, N, S> Forward<'a, Matrix<N, Dynamic, U1, S>, DVector<Complex<N>>> for FFT<N>
    where
        N : Scalar + Debug + Copy + From<f32> + RealField,
        Complex<N> : Scalar + Clone,
        S : ContiguousStorage<N, Dynamic, U1>
{

    /*fn forward(mut self, src : &'a DVector<N>) -> DVector<Complex<N>> {
        self.plan.forward(&s);
        self.plan.forward_buffer
    }*/

    fn forward_from(&'a mut self, s : &Matrix<N, Dynamic, U1, S>) -> &'a DVector<Complex<N>> {
        // let s : DVectorSlice<'a, N> = s.into();
        /*let ans =*/ self.plan.forward(&s);
        &self.plan.forward_buffer
        // self.back_call = false;
        // &(*ans)
    }

    /*fn partial_backward<S>(&'a mut self, n : usize) -> DVectorSlice<'a, N> {
        // let mut dst = self.plan.take().unwrap();
        // plan.backward_to(&mut dst);
        // self.dst = Some(dst);
        // dst.as_ref().into()
        unimplemented!()
    }*/

    fn coefficients(&'a self) ->  &'a DVector<Complex<N>> {
        &self.plan.forward_buffer
    }

    fn coefficients_mut(&'a mut self) ->  &'a mut DVector<Complex<N>> {
        &mut self.plan.forward_buffer
    }

    /*fn domain(&'a self) -> Option<&'a DVector<N>> {
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
    }*/

}

impl<'a, N, /*S*/ > Backward<'a, DVector<N>, DVector<Complex<N>>> for FFT<N>
    where
        N : Scalar + Debug + Copy + From<f32> + RealField,
        Complex<N> : Scalar + Clone,
        //S : ContiguousStorageMut<N, Dynamic, U1>
{

    fn backward_from(&'a mut self, coefs : &'a DVector<Complex<N>>) -> Option<&'a DVector<N>> {
        self.plan.forward_buffer.copy_from(&coefs);
        self.backward_from_self()
    }

    fn backward_from_self(&'a mut self) -> Option<&'a DVector<N>> {
        if let Some(mut dom) = self.take_domain() {
            self.backward_mut(&mut dom);
            self.domain = Some(dom);
            self.domain()
        } else {
            None
        }
    }

    fn backward_mut(&self, dst : &mut DVector<N>) {
        // let mut b_buf = self.domain.take().unwrap();
        self.plan.backward_to(dst);
        // self.domain = Some(b_buf);
        // self.back_call = true;
        // self.domain.as_ref().unwrap()
    }

    fn take_domain(&mut self) -> Option<DVector<N>> {
        self.domain.take()
    }

    fn domain(&'a self) -> Option<&'a DVector<N>> {
        self.domain.as_ref()
    }

    fn domain_mut(&'a mut self) -> Option<&'a mut DVector<N>> {
        self.domain.as_mut()
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
    // back_call : bool
}

#[cfg(feature = "mkl")]
impl FFT2D<f32> {

    pub fn new<S : Into<DMatrix<f32>>>(s : S) -> Self {
        let domain : DMatrix<f32> = s.into();
        let mut plan = FFTPlan::new((domain.nrows(), domain.ncols())).unwrap();
        plan.forward(&domain);
        // let fft = Self{ plan, domain : Some(domain), back_call : true };
        // fft
        Self{ plan, domain : Some(domain) }
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
        //let fft = Self{ plan, domain : Some(domain), back_call : true };
        //fft
        Self{ plan, domain : Some(domain) }
    }

    pub fn new_empty(nrow : usize, ncol : usize) -> Self {
        Self::new(DMatrix::from_element(nrow, ncol, 0.0))
    }
}

#[cfg(feature = "mkl")]
impl<'a, N, S> Forward<'a, Matrix<N, Dynamic, Dynamic, S>, DMatrix<Complex<N>>> for FFT2D<N>
    where
        N : Scalar + Debug + Copy + From<f32> + RealField,
        Complex<N> : Scalar + Clone,
        S : ContiguousStorage<N, Dynamic, Dynamic>
{

    /*fn forward(mut self, src : &'a DVector<N>) -> DVector<Complex<N>> {
        self.plan.forward(&s);
        self.plan.forward_buffer
    }*/

    fn forward_from(&'a mut self, s : &Matrix<N, Dynamic, Dynamic, S>) -> &'a DMatrix<Complex<N>> {
        // let s : Matrix<N, Dynamic, Dynamic, SliceStorage<'a, N, Dynamic, Dynamic, U1, Dynamic>> = s.into();
        // let ans =
        self.plan.forward(&s);
        &self.plan.forward_buffer
        // self.back_call = false;
        // &(*ans)
    }

    //fn partial_backward<S>(&'a mut self, n : usize) -> DMatrixSlice<'a, N> {
    //    unimplemented!()
    //}

    fn coefficients(&'a self) -> &'a DMatrix<Complex<N>> {
        &self.plan.forward_buffer
    }

    fn coefficients_mut(&'a mut self) -> &'a mut DMatrix<Complex<N>> {
        &mut self.plan.forward_buffer
    }

    /*fn domain(&'a self) -> Option<&'a DMatrix<N>> {
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
    }*/

}

#[cfg(feature = "mkl")]
impl<'a, N, /*S*/ > Backward<'a, DMatrix<N>, DMatrix<Complex<N>>> for FFT2D<N>
    where
        N : Scalar + Debug + Copy + From<f32> + RealField,
        Complex<N> : Scalar + Clone,
        //S : ContiguousStorageMut<N, Dynamic, Dynamic>
{

    fn backward_from(&'a mut self, coefs : &'a DMatrix<Complex<N>>) -> Option<&'a DMatrix<N>> {
        self.plan.forward_buffer.copy_from(&coefs);
        self.backward_from_self()
    }

    fn backward_from_self(&'a mut self) -> Option<&'a DMatrix<N>> {
        if let Some(mut dom) = self.take_domain() {
            self.backward_mut(&mut dom);
            self.domain = Some(dom);
            self.domain()
        } else {
            None
        }
    }

    fn backward_mut(&self, dst : &mut DMatrix<N>) {
        // let mut b_buf = self.domain.take().unwrap();
        self.plan.backward_to(dst);
        // self.domain = Some(b_buf);
        // self.back_call = true;
        //self.domain.as_ref().unwrap()
    }

    fn take_domain(&mut self) -> Option<DMatrix<N>> {
        self.domain.take()
    }

    fn domain(&'a self) -> Option<&'a DMatrix<N>> {
        self.domain.as_ref()
    }

    fn domain_mut(&'a mut self) -> Option<&'a mut DMatrix<N>> {
        self.domain.as_mut()
    }

}*/


