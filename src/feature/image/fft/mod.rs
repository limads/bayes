use nalgebra::*;
use super::*;
use nalgebra::storage::*;
use simba::scalar::RealField;
use std::fmt::Debug;
use super::*;
use crate::feature::signal::ops::*;
use crate::feature::signal::fft::mkl::*;

pub struct Fourier2D<N> 
where
    N : Scalar,
    Complex<N> : Scalar
{
    plan : FFTPlan<N>
}

impl Fourier2D<f32> {

    pub fn new(sz : (usize, usize)) -> Result<Self, String> {
        Ok(Self { plan : FFTPlan::<f32>::new(sz)? })
    }
}

impl Fourier2D<f64> {

    pub fn new(sz : (usize, usize)) -> Result<Self, String> {
        Ok(Self { plan : FFTPlan::<f64>::new(sz)? })
    }
    
}

impl<N> Forward<Image<N>> for Fourier2D<N> 
where
    N : Scalar + From<f32>,
    Complex<N> : Scalar
{
    
    type Output = Image<Complex<N>>;
    
    fn forward_mut(&self, src : &Image<N>, dst : &mut Self::Output) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    fn forward(&self, src : &Image<N>) -> Self::Output {
        let zero = N::from(0.0 as f32);
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, Complex::new(zero.clone(), zero));
        self.forward_mut(src, &mut dst);
        dst
    }
}

impl<N> Backward<Image<Complex<N>>> for Fourier2D<N> 
where
    N : Scalar + From<f32>,
    Complex<N> : Scalar
{
    
    type Output = Image<N>;
    
    fn backward_mut(&self, src : &Image<Complex<N>>, dst : &mut Self::Output) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    fn backward(&self, src : &Image<Complex<N>>) -> Self::Output {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, N::from(0.0 as f32));
        self.backward_mut(src, &mut dst);
        dst
    }
}

/*/// If the FFT is built with a window, perform the short-time fourier transform.
/// The transform yields an interator over separate windows. If the FFT is built
/// without a window, the iterator yield a single element. The second element is
/// an overlap, informing if the window should be over contiguous or overlapping
/// signal segments (which is necessary if you are studying transitions that are
/// too large relative to the window and might be missed if they are over window
/// transitions). Each transition point is a categorical (or bernoulli for binary
/// transitions) with a dirichlet (beta) prior.
/// Self will decompose the signal at windows of size len.
/// Signal is shift-invariant within windows at the cost
/// of reduced spatial resolution. Larger window sizes
/// increase spatial resolution at each window at the cost
/// of not being able to examine short-scale temporal
/// changes. After setting the window, take FFT only of the
/// updated window, leaving past data at their old state.
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
    N : Scalar + Into<f64>,
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
}*/

