use nalgebra::*;
use nalgebra::storage::*;

#[cfg(feature = "mkl")]
pub mod fft;

pub mod dwt;

pub mod conv;

/// Generic trait for frequency or spatio/temporal-frequency domain transformations.
/// The input data to the try_forward/forward methods must be a matrix or vector
/// of scalar type M; The input data to the try_backward_to/backward_to methods
/// must be a matrix or vector of scalar type N. The type C prescribes the dimensionality
/// of the input/output pair (either vector or matrix). The implementor is
/// assumed to own the forward transform buffer (i.e. the frequency domain data), which
/// is why the forward methods return a reference with lifetime tied to the implementor.
/// The backward buffer (if needed) is assumed to have been pre-allocated by the user, which should opt
/// to use the try_backward_to directly (which takes a mutable reference and returns it in case of success)
/// or the try_backward, which takes ownership of the buffer and returns it to the user in case of
/// success. Implementors should worry only about implementing the try_forward and try_backward_to
/// calls.
pub trait FrequencyBasis<M, N, C>
    where
        M : Scalar,
        N : Scalar,
        C : Dim
{

    fn try_forward<'a, S>(
        &'a mut self,
        src : &Matrix<M,Dynamic,C,S>
    ) -> Option<&'a mut Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>>>
        where
            S : ContiguousStorage<M, Dynamic, C>;

    fn try_backward_to<'a, 'b, /*S,*/ SB>(
        &'a self,
        //src : &'a Matrix<N,Dynamic,C,S>,
        dst : &'b mut Matrix<M,Dynamic,C,SB>
    ) -> Option<&'b mut Matrix<M,Dynamic,C,SB>>
        where
            //S : ContiguousStorage<N, Dynamic, C>,
            SB : ContiguousStorageMut<M, Dynamic, C>;
            //for<'c> Self : 'a ;

    fn try_backward<'a, /*S,*/ SB>(
        &'a self,
        //src : &'a Matrix<N,Dynamic,C,S>,
        mut dst : Matrix<M, Dynamic, C, VecStorage<M, Dynamic, C>>
    ) -> Option<Matrix<M,Dynamic,C,VecStorage<M, Dynamic, C>>>
        where
            //S : ContiguousStorage<N, Dynamic, C>,
            SB : ContiguousStorage<M, Dynamic, C>,
            VecStorage<M, Dynamic, C> : ContiguousStorage<M, Dynamic, C>
    {
        self.try_backward_to( /*&src,*/ &mut dst)?;
        Some(dst)
    }

    fn forward<'a, S>(
        &'a mut self,
        src : &Matrix<M,Dynamic,C,S>
    ) -> &'a mut Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>>
        where
            S : ContiguousStorage<M, Dynamic, C>
    {
        self.try_forward(src).unwrap()
    }

    fn backward_to<'a, 'b, /*S,*/ SB>(
        &'a self,
        //src : &'a Matrix<N,Dynamic,C,S>,
        dst : &'b mut Matrix<M,Dynamic,C,SB>
    ) -> &'b mut Matrix<M,Dynamic,C,SB>
        where
            //S : ContiguousStorage<N, Dynamic, C>,
            SB : ContiguousStorageMut<M, Dynamic, C>,
            //Self: 'b
    {
        self.try_backward_to( /*src,*/ dst).unwrap()
    }

    /// Applies f to the forward transform of src, then transform
    /// back to the original domain.
    fn apply_forward<S, SD, SF, F>(
        &mut self,
        f : F,
        src : &Matrix<M,Dynamic,C,S>,
        mut dst : Matrix<M, Dynamic, C, SD>
    ) -> Option<Matrix<M,Dynamic,C,SD>>
        where
            S : ContiguousStorage<M, Dynamic, C>,
            SD : ContiguousStorageMut<M, Dynamic, C>,
            F : Fn(&mut Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>>)->Option<()>,
    {
        self.try_forward(src).and_then(|fwd| f(fwd) )?;
        //let fwd = f(ans)?;
        self.try_backward_to( /*&fwd,*/ &mut dst)?;
        Some(dst)
    }

}


