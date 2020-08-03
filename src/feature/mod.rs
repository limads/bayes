use nalgebra::*;
use crate::sample::Sample;
use nalgebra::sparse::CsMatrix;
use crate::distr::*;

// Linear discriminant analysis
// pub mod lda;

// Dissimilarity metrics and related clustering and matching algorithms
pub mod metric;

/// Multiresolution analysis and signal processing utilities
pub mod mra;

// Principal components analysis basis reduction
//pub mod pca;

// Polynomial and spline basis expansion
// pub mod poly;

mod extraction;

pub use extraction::*;

// use metric::Metric;

/// A feature is a sparse subset of the coefficients or basis after an expansion
/// (Fourier, Wavelet PCA or LDA) is applied. Which elements are preserved is
/// determined by some feature extraction rule, which might involve preserving some
/// ordered subset, subsampling, or a combination of the two rules. Coefficients are
/// represented in a Sample implementor, so they can be input to probabilistic models.
///
/// A lossy version of the signal might be built by using the data at the indices and coefficients
/// columns to build an approximated frequency/spatial-frequency matrix via the sparse matrix
/// (DWT) or interpolation over indices (FFT). Coefficients with complex entries
/// are separated into an absolute value real component and a phase real component.
///
/// More generally, Feature are (serializable) feature extraction algorithms that take an arbitrary
/// structure (such as a stream, text or image) and output a Sample implementor
/// (exchangeable observation container that is sortable, splittable, and samplable). From this
/// implementor, the original structure can be built, at least in an approximated fashion. This
/// constraint allows an automatic way to build generative models: Walk the pipeline backwards,
/// sampling from the model and using the samples to re-build the signal.
pub trait Feature
// where
//    Self : Invert<Self::Output>
{

    type Input;

    type Output;

    /// Returns a sample with the coefficients/observations satisfying
    /// the informed criteria. Individual coefficients/observations are
    /// considered to be the sampling units in this context. Might return
    /// zero samples if criteria is not met. Indices are the first columns,
    /// the coefficient value the last column. Different signals might return
    /// wildly differently-sized matrices depending on the criteria used.
    fn extract(sig : &Self::Input) -> Self::Output;

    /// For each informed signal, preserve only the n coefficients with
    /// largest absolute value. Order the coefficients by absolute value,
    /// and return one row per signal sample. Here, individual signals are
    /// the sampling units, and the ordered coefficients (ix1..ixn..value)
    /// are arranged as rows. If at least one signal does not yield the required
    /// number of coefficients, this function return no samples. If all samples
    /// have the required number of signals, all samples will have the same
    /// number of columns (unlike extract), but columns will not in general
    /// represent the same coefficient (only the relative ordering of the coefficient
    /// wrt. the criterion adopted).
    fn preserve(sigs : &[&Self::Input], n : usize) -> Self::Output;

}

/// Invert should be implemented by feature extraction algorithms to
/// allow the automatic formation of full generative models. If the user
/// builds a pipeline only with Invert implementors, sampling from a probabilistic
/// graph propagates the data up to the generation of a full signal or image
/// from which the model was fit. Invert is also implemented by linear operations
/// (multiply-add), conversions (floating-point to u8 for example) and
/// LTI operations (subsampling/upsampling, convolution).
pub trait Invert<S>
where {

    fn invert(feat : Self) -> S;

}

pub enum PipelineError {
    Step(usize),
    Extract,
    Sample(usize)
}

pub enum Step<D, I>
where
    I : Invert<D>
{
    Generic(Box<dyn FnMut(D)->Option<D>>),
    Invertible(I)
}

/*impl<D, I> From<F> for Step<D, I>
where
    F : FnMut(D)->Option<D>,
    I : Invert<D>
{

    fn from(f : F) {
        Self::Generic(Box::new(f))
    }
}

impl<D, I> From<I> for Step<D, I>
where
    I : Invert<D>,
{
    fn from(f : F) {
        Self::Generic(Box::new(f))
    }

}*/

/// Encapsulates a feature extraction pipeline. A pipeline is built with the builder
/// pattern using a series of transform(.) calls, which build the pipeline, ending
/// at a extract(.) call, which extracts a Sample implementor that can interface
/// with the probabilistic graph, set at the .model(.) call.
///
/// A pipeline is used to build simple, serial transformation
/// steps that end in a sample of exchangeable observations O that can be modelled. Alternatively, a pipeline
/// can execute many feature extraction steps via the accumulate() method, extending the
/// sample implementor taken at the end. This is useful if you have a series of images
/// or temporal streams and want to accumulate their features before running a probabilistic
/// model.
///
/// The library offers several utilities under the signal:: module that
/// can be directly plugged-in as parameters to those functions.
///
/// Pipelines may contain generative models: if a model is present at the
/// last element, and all operations implement Invert, a signal can be reconstructed
/// by sampling from the model and walking the graph backwards.
///
/// # Example
///
/// ```
/// use bayes::feature::Pipeline;
/// use bayes::mra::{signal::*, basis::DWT};
///
/// let dwt = DWT2D::new(16);
/// let pipe = Pipeline::create(16, 16)
///     .convert(signal::subsample(2))
///     .transform(signal::convolve(&[1.,2.]))
///     .transform(|s| dwt.forward(&s))
///     .extract(local::maxima)
///     .model(Normal::new());
///
/// let ans = pipe.process(&[1,2,3]).expect("Pipeline failed);
/// let ans = pipe.generate(None);
/// model.fit(&ans);
/// ```
pub struct Pipeline<D, S, E>
where
    //E : Estimator<P>
    //P : Distribution
{

    /// Will hold Step enumeration with either &dyn Invert
    /// or &dyn FnMut(D)->D
    transf : Vec<Box<dyn FnMut(D)->Option<D>>>,

    /// Returns a random and constant sample component.
    /// Will hold &dyn Feature
    extract : Option<Box<dyn FnMut(D)->(Option<S>, Option<S>)>>,

    dim : (usize, usize),

    acc : Vec<(S, Option<S>)>,

    distr : Option<E>
}

impl<D, S, E> Pipeline<D, S, E>
where
    //E : Estimator<P>,
    //P : Distribution
{

    pub fn create(dim : (usize, usize)) -> Self {
        Self{ transf : Vec::new(), extract : None, dim, acc : Vec::new(), distr : None }
    }

    /// Extends the pipeline by adding an arbitrary
    /// signal-to-signal transformation
    pub fn transform<F>(mut self, f : F) -> Self
    where
        F : FnMut(D)->Option<D> + 'static
    {
        self.transf.push(Box::new(f));
        self
    }

    /// Extends the pipeline by adding a final
    /// signal-to-sample transformation (feature extraction).
    pub fn extract<F>(mut self, f : F) -> Self
    where
        F : FnMut(D)->(Option<S>, Option<S>) + 'static
    {
        self.extract = Some(Box::new(f));
        self
    }

    /// Executes the pipeline, returning the final feature sample
    /// if successful, without accumulating it.
    /// Indicates which element of the pipeline failed
    /// in case of error.
    pub fn process(&mut self, mut d : D) -> Result<(S, Option<S>), PipelineError> {
        for (i, func_s) in self.transf.iter_mut().enumerate() {
            d = (func_s)(d).ok_or(PipelineError::Step(i))?;
        }
        let func_e = self.extract.as_mut().ok_or(PipelineError::Extract)?;
        let (opt_y, opt_x) = (func_e)(d);
        let y = opt_y.ok_or(PipelineError::Extract)?;
        Ok((y, opt_x))
    }

    pub fn accumulate(&mut self, d : D) -> Result<(), PipelineError> {
        let ans = self.process(d)?;
        self.acc.push(ans);
        Ok(())
    }

    pub fn take_data(mut self) -> Vec<(S, Option<S>)> {
        self.acc
    }

    pub fn model(mut self, e : E) -> Self {
        self.distr = Some(e);
        self
    }

    // Pass generic sample implementor here.
    pub fn generate(&self, x : Option<DMatrix<f64>>) -> Option<D> {
        // Sample from posterior if available
        // Iterate transforms back calling invert if available.
        // Output result.
        unimplemented!()
    }

    pub fn fit(&self, data : &[D]) -> Result<(), PipelineError> {
        /*let acc = self.accumulate()?;
        for (i, ans) in self.acc.iter() {
            self.distr.fit(&s)
                .map_err(|_| PipelineError::Sample(i))?;
        }
        Ok(())*/
        unimplemented!()
    }

    pub fn view_estimator(&self) -> Option<&E> {
        self.distr.as_ref()
    }

    pub fn take_estimator(mut self) -> Option<E> {
        self.distr
    }

    //pub fn take_posterior(mut self) -> Option<P> {
    //    self.take_estimator().map(|e| e.take_posterior() )
    //}

}

/*pub trait Constant
where
    Self : Sample
{


}

pub enum Data<S, O, G>
where
    S : Sample<O>
    G : Signal
{

    /// Carries a sample implementor with a split point
    Sample(s : S<O>, usize),

    /// A pipeline yields a sample.
    Pipe(p : Pipeline<G>)

}

pub trait Node {

    pub fn children
    /// Sends a message to child nodes.
    pub fn propagate(d : Data) -> Data {

    }

    /// Receives a message
    pub fn receive(d : Data) -> Data {

    }
}

#[test]
fn build() {
    let graph = Node::build()
        .add(Pipeline::new((100,1))
            .transform(Resample::Subsample{ 10 })
            .transform(Offset::new(1))
            .extract(Identity))
        .add(Normal::new(10, None, None));
}*/
