use super::iter::*;
use nalgebra::*;

pub trait Convolve {

    fn convolve(&self, kernel : &Self) -> Self;

}

impl<N> Convolve for DMatrix<N>
    where
        N : Scalar + std::ops::Mul + simba::scalar::ClosedMul<N> +
            simba::scalar::Field + simba::scalar::SupersetOf<f64>,
        DMatrix<N> : WindowIterate<N, VecStorage<N, Dynamic, Dynamic>>
{

    fn convolve(&self, kernel : &Self) -> Self {
        self.windows(kernel.shape())
            .pool(|win| win.component_mul(&kernel).sum())
    }
}
