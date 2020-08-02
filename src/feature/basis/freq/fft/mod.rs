use nalgebra::*;
use super::*;
use nalgebra::storage::*;
use simba::scalar::RealField;
use std::fmt::Debug;
use crate::feature::basis::*;
use crate::signal::conv::*;

/// Wrappers over MKL FFT routines.
pub mod mkl;

use mkl::*;

pub enum Overlap {
    None,
    Half,
    Quarter
}

/// If the FFT is built with a window, perform the short-time fourier transform.
/// The transform yields an interator over separate windows. If the FFT is built
/// without a window, the iterator yield a single element. The second element is
/// an overlap, informing if the window should be over contiguous or overlapping
/// signal segments (which is necessary if you are studying transitions that are
/// too large relative to the window and might be missed if they are over window
/// transitions). Each transition point is a categorical (or bernoulli for binary
/// transitions) with a dirichlet (beta) prior.
pub enum Window {

    /// The box window simply tiles the temporal/spatial
    /// domain and perform the Fourier transform separately
    /// for each segment.
    Box(usize, Overlap),

    /// The Hanning window tiles the temporal/spatial
    /// domain and pre-multiplies the signal with the
    /// a box-like window, but that is attenuated with
    /// a cosine decay function at the borders to guarantee
    /// neighboring segments are continuous on the circular domain.
    Hanning(usize, Overlap),

    /// A gabor window pre-multiply the signal with a
    /// gaussian before applying the FFT. The effect is
    /// the same of taking the dot-product of the signal
    /// with separate, localized gabor functions (complex exponentials
    /// which don't quite decay to zero but get very close to).
    Gabor(usize, Overlap)

}

struct WindowContent<N, C>
where
    N : Scalar + Into<f64>
    C : Dim
{
    win : Window,
    buf : Option<DMatrix<N, Dynamic, C, VecStorage<N, Dynamic, C>>>
}

impl<N> WindowContent<N, U1>
where
    N : Scalar + Into<f64> + From<f32>
{

    pub fn new(win : Window) -> Self {
        match &win {
            Window::Box(n, _) => {
                Self{ buf : None, win }
            },
            Window::Hanning(n, _) => {
                Self{ buf : Some(hann(n)), win }
            },
            Window::Gabor(n, _) => {
                unimplemented!()
            }
        }
    }

}

impl<N> WindowContent<N, Dynamic>
where
    N : Scalar + Into<f64> + From<f32>
{

    pub fn new(win : Window) -> Self {
        match &win {
            Window::Box(n, _) => {
                Self{ buf : None, win }
            },
            Window::Hanning(n, _) => {
                Self{ buf : hann2(n), win }
            },
            Window::Gabor(n, _) => {
                unimplemented!()
            }
        }
    }

}

impl<N, C> WindowContent<N, C>
    where
    N : Scalar + Into<f64> + From<f32>
    C : Dim
{

    pub fn apply(
        signal : &Matrix<f64, Dynamic, C, S>
    ) -> Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>
    where
        S : Storage<f64, Dynamic, C>,
        Matrix<f64, Dynamic, C, S> : WindowIterate<N, S>
    {

        match self.overlap {
            Self::Box(n, overlap) => panic!("Box window does not require re-allocation"),
            Self::Hanning(n, overlap) | Self::Gabor(n, overlap) => {
                let step_sz = match overlap {
                    Overlap::None => 1,
                    Overlap::Half => n / 2,
                    Overlap::Quarter => n / 4
                };
                // TODO extend this matrix rows/cols if signal has overlap
                let mut dst = signal.clone();
                for (win, i) in signal.windows().step_by(step_sz).enumerate() {
                    win.component_mul_to(&self.buf.unwrap(), dst.slice_mut(pos));
                }
                dst
            }
        }
    }
}

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



