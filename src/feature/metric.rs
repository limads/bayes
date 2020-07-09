use crate::sample::Sample;
use nalgebra::*;
use nalgebra::sparse::CsMatrix;

/// A metric is a scalar value that captures the distance or dissimilarity between
/// a pair of n-dimensional observations. If those observations are arranged into
/// a Sample implementor, within(.) returns the square matrix of pair-wise metrics
/// of all observations within the sample. Metrics can also be calculated as the
/// pair-wise comparison of all elements of a pair of sample implementors via the between(.)
/// method.
pub trait Metric
where Self : Sized
{

    fn within<S>(a : S) -> Self
    where S : Into<DMatrix<f64>> + Clone
    {
        Self::between(a.clone(), a)
    }

    fn between<S>(a : S, b : S) -> Self
    where S : Into<DMatrix<f64>>;

    /// Returns the full distance matrix. For within-sample comparisons,
    /// (Self::within), the diagonal values will always be zero, since the
    /// elements are being compared to themselves at the i==j entries.
    fn full(&self) -> DMatrix<f64>;

    /// Return a sparse matrix containing the n-smallest distances.
    /// The row index correspond to the observation index at left set;
    /// the column index correspond to the observation index at the
    /// right set. The implementor should guarantee that if a pair-wise
    /// comparison is being made (Self::within), the comparison of an
    /// element with itself is not returned (the sparse matrix will never
    /// have an element in the diagonal).
    fn closest(&self, n : usize) -> CsMatrix<f64>;
}

/// Represents an upper triangular matrix of euclidian distances
pub struct Euclidian {

}

pub struct Manhattan {

}

/// Represents an upper triangular matrix of Manhattan distances
impl Metric for Manhattan {

    fn between<S>(a : S, b : S) -> Self
        where S : Into<DMatrix<f64>>
    {
        unimplemented!()
    }

    fn full(&self) -> DMatrix<f64> {
        unimplemented!()
    }

    fn closest(&self, n : usize) -> CsMatrix<f64> {
        unimplemented!()
    }

}

impl Metric for Euclidian {

    fn between<S>(a : S, b : S) -> Self
        where S : Into<DMatrix<f64>>
    {
        unimplemented!()
    }

    fn full(&self) -> DMatrix<f64> {
        unimplemented!()
    }

    fn closest(&self, n : usize) -> CsMatrix<f64> {
        unimplemented!()
    }

}


