use nalgebra::*;
use super::*;
use nalgebra::storage::*;
use simba::scalar::RealField;
use std::fmt::Debug;
use super::*;
use crate::feature::signal::fft::mkl::*;
use num_traits::Num;
use simba::scalar::SubsetOf;

/// Two-dimensional Fourier transform
pub struct Fourier2D<N> 
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + Num + From<f32> + SimdPartialOrd,
    f64 : SubsetOf<N>,
    Complex<N> : Scalar
{
    plan : FFTPlan<N>
}

impl Fourier2D<f32> {

    pub fn new(nrow : usize, ncol : usize) -> Result<Self, String> {
        Ok(Self { plan : FFTPlan::<f32>::new((nrow, ncol))? })
    }
}

impl Fourier2D<f64> {

    pub fn new(nrow : usize, ncol : usize) -> Result<Self, String> {
        Ok(Self { plan : FFTPlan::<f64>::new((nrow, ncol))? })
    }
    
}

impl<N> Fourier2D<N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + Num + From<f32> /*+ SimdPartialOrd*/,
    // Complex<N> : Scalar,
    //Complex<N> : SimdComplexField<SimdRealField=N>,
    // Complex<N> : SimdComplexField,
    N : RealField,
    f64 : SubsetOf<N>
{

    pub fn forward_mut(&self, src : &Image<N>, dst : &mut ImageSpectrum<N>) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    pub fn forward(&self, src : &Image<N>) -> ImageSpectrum<N> {
        let zero = N::from(0.0 as f32);
        let (nrows, ncols) = self.plan.shape();
        let mut dst = ImageSpectrum::new_constant(nrows, ncols, Complex::new(zero.clone(), zero));
        self.forward_mut(src, &mut dst);
        dst
    }
    
    pub fn backward_mut(&self, src : &ImageSpectrum<N>, dst : &mut Image<N>) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    pub fn backward(&self, src : &ImageSpectrum<N>) -> Image<N> {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, N::from(0.0 as f32));
        self.backward_mut(src, &mut dst);
        dst
    }
}

/*impl<N> Forward<Image<N>> for Fourier2D<N> 
where
    N : Scalar + From<f32> + Num,
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
    N : Scalar + From<f32> + Num,
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
}*/

/// Output of a two-dimensional Fourier transform.
#[derive(Clone, Debug)]
pub struct ImageSpectrum<N> 
where
    N : Scalar + Copy
{
    buf : DMatrix<Complex<N>>
}

/// Slice over an image spectrum.
pub struct ImageBand<'a, N> 
where
    N : Scalar + Copy
{
    s : DMatrixSlice<'a, N>
}

impl<'a, N> ImageSpectrum<N> 
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + RealField,
    Complex<N> : ComplexField<RealField=N>
    // Complex<N> : SimdComplexField<SimdRealField=N>,
    // bool : SimdBool
    // N : RealField
{

    pub fn new_constant(nrows : usize, ncols : usize, value : Complex<N>) -> Self {
        Self{ buf : DMatrix::from_element(nrows, ncols, value) }
    }
    
    pub fn iter(&self) -> impl Iterator<Item=&Complex<N>> {
        self.buf.iter()
    }
    
    pub fn iter_mut(&'a mut self) -> impl Iterator<Item=&'a mut Complex<N>> {
        self.buf.iter_mut()
    }

    pub fn conjugate_mut(&mut self) {
        self.buf.conjugate_mut()
    }
    
    pub fn scale_by(&mut self, scalar : N)  {
        self.buf.scale_mut(scalar);
    }
    
    pub fn unscale_by(&mut self, scalar : N)  {
        self.buf.unscale_mut(scalar);
    }
    
    pub fn component_scale(&mut self, other : &ImageSpectrum<N>) {
        self.buf.component_mul_mut(&other.buf);
    }
    
    //pub fn bands(&self) -> impl Iterator<Item=ImageBand> {
    //} 
    
}

impl<N> AsRef<[Complex<N>]> for ImageSpectrum<N> 
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &[Complex<N>] {
        self.buf.data.as_slice()
    }
}

impl<N> AsMut<[Complex<N>]> for ImageSpectrum<N> 
where
    N : Scalar + Copy
{
    fn as_mut(&mut self) -> &mut [Complex<N>] {
        self.buf.data.as_mut_slice()
    }
}

impl<N> AsRef<DMatrix<Complex<N>>> for ImageSpectrum<N> 
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &DMatrix<Complex<N>> {
        &self.buf
    }
}

impl<N> From<DMatrix<Complex<N>>> for ImageSpectrum<N> 
where
    N : Scalar + Copy
{
    fn from(s : DMatrix<Complex<N>>) -> Self {
        Self{ buf : s }
    }
}

/*impl<N> From<Vec<N>> for ImagePyramid<N> 
where
    N : Scalar
{
    fn from(s : Vec<N>) -> Self {
        Self{ buf : DVector::from_vec(s) }
    }
}*/

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

