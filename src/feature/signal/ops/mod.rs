use nalgebra::*;
use nalgebra::storage::*;

pub trait Forward<D> {

    type Output;
    
    fn forward_mut(&self, src : &D, dst : &mut Self::Output);
    
    fn forward(&self, src : &D) -> Self::Output;
}

pub trait Backward<D> {

    type Output;
    
    fn backward_mut(&self, src : &D, dst : &mut Self::Output);
    
    fn backward(&self, src : &D) -> Self::Output;
    
}

/*
nalgebra conventions:

<algorithm>.<opname>(&input) -> owned_output Takes a reference to self and a reference to the input and
provides a newly-allocated output (perhaps cloning input and applying the op in-place to the cloned data)

<algorithm>.<opname_mut>(&mut output) Takes a reference to self (which holds the domain) and outputs
the result to the informed pre-allocated buffer.

Use Option for fallible operations because usually those algorithms can fail in a single
way (buffer size is not sufficient).

The difference is that instead of returning owned_output, we return &output for the operations,
since we always assume the existence of a pre-allocated buffer inside the structure that is the
destination of the operation.

We add the following convention: forward_from(.) and backward_from(.) modify the internal
buffers from an external reference.
*/

/*/// Implemented by forward-transform algorithms (FFT, DWT), which modify
/// their internal state by calculating the forward transform from src,
/// and which allows reading the updated coefficients via coefficients(.)
/// and coefficients_mut(.). The coefficients are stored at a generic container
/// C, defined by the implementation.
pub trait Forward<'a, D, C>
where
    Self : Sized + 'a,
    // C : Clone + 'a
{

    /// Apply the forward transform from buffer src into Self, updating its coefficients.
    fn forward_from(&'a mut self, src : &D) -> &'a C;

    /// Iterate over eigenvalues of the decomposition (PCA/LDA) or over
    /// complex coefficients at a vector or matrix (FFT) or over the real
    /// coefficient windows (DWT). Returns single result for PCA/LDA (Option).
    /// Return groups of coefficients for splines, depending on the region
    /// the spline is centered at (Vec).
    fn coefficients(&'a self) -> &'a C;

    fn coefficients_mut(&'a mut self) -> &'a mut C;
    
    // This should consume the structure and return its coefficients.
    // fn take_coefficients(self) -> C;

}

/// Implemented by backward-transform algorithms (IFFT,IDWT),
/// which modify the destination buffer from the inverse transform
/// applied to the data carried into Self. The original domain is written
/// into a generic container D, defined by the implementation.
pub trait Backward<'a, D, C>
where
    Self : Forward<'a, D, C> + Sized
{

    fn backward_from(&'a mut self, coefs : &'a C) -> Option<&'a D>;

    fn backward_from_self(&'a mut self) -> Option<&'a D>;

    fn take_domain(&mut self) -> Option<D>;

    fn domain(&'a self) -> Option<&'a D>;

    fn domain_mut(&'a mut self) -> Option<&'a mut D>;

    /// Apply the backward transform into original buffer dst.
    fn backward_mut(&'a self, dst : &'a mut D);

    // fn backward(&self) -> D {
    //    let mut dom = self.domain().clone();
    //    self.backward_mut(&mut dom);
    //    dom
    // }

}*/


