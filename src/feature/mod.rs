use nalgebra::*;
use crate::sample::Sample;
use nalgebra::sparse::CsMatrix;
use basis::Basis;

// Clustering algorithms.
// pub mod cluster;

/// Basis expansion and reduction techniques useful to model non-linear and non-stationary
/// processes or high-dimensional data. Functionality is offered moslty via bindings to
/// GSL and MKL (work in progress).
pub mod basis;

// Polynomial basis expansions (Splines).
// pub mod poly;

/// Traits and structures for similarity and dissimilarity metrics.
pub mod metric;

use metric::Metric;

/// A feature is a sparse subset of the coefficients or basis after an expansion
/// (Fourier, Wavelet PCA or LDA) is applied. Which elements are preserved is
/// determined by some feature extraction rule, which might involve preserving some
/// ordered subset, subsampling, or a combination of the two rules.
pub trait Feature<'a, B, M, N, C>
where
    B : Basis<'a, M, N, C>,
    M: Scalar,
    N : Scalar,
    C : Dim
{

    fn view_features(&self) -> CsMatrix<N>;

    fn extract(&mut self, b : &B) -> CsMatrix<N>;

    fn match_with<T>(&self, other : &Self) -> T
        where T : Metric
    {
        //T::between(self.view_features(), other.view_features())
        unimplemented!()
    }

}

//impl<B> Into<DMatrix<f64>> for Feature<B> {
//}

// impl<B> Sample for Feature<B> {
// }

/// A match represents the result of the matching process
/// between a pair of features. This match carries both
/// a metric and extra parameters that quality the match,
/// such as a global relative translation or scale.
pub struct Match<M>
    where
        M : Metric
{
    m : M
}

// impl<M> Sample for Match<M> {
// }


