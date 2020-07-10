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

    fn dim(&self) -> usize;

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

    /// Return a matrix with the n distances closest to the informed point.
    /// The row index correspond to the observation index at left set;
    /// the column index correspond to the observation index at the
    /// right set. The implementor should guarantee that if a pair-wise
    /// comparison is being made (Self::within), the comparison of an
    /// element with itself is not returned (the sparse matrix will never
    /// have an element in the diagonal).
    fn closest_to(&self, pt : &[f64], n : usize) -> CsMatrix<f64>;

    /// Return a sparse matrix containing the n-smallest distances.
    fn smallest(&self, n : usize) -> CsMatrix<f64> {
        let mut pt : Vec<f64> = Vec::new();
        pt.extend((0..self.dim()).map(|_| 0.0 ));
        self.closest_to(&pt[..], n)
    }

    // TODO add provided method for histogram.
}

/// Represents an upper triangular matrix of euclidian distances
pub struct Euclidian {
    dim : usize,
    dst : DMatrix<f64>
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

    fn dim(&self) -> usize {
        unimplemented!()
    }

    fn closest_to(&self, pt : &[f64], n : usize) -> CsMatrix<f64> {
        unimplemented!()
    }

}

impl Metric for Euclidian {

    fn dim(&self) -> usize {
        self.dim
    }

    fn between<S>(a : S, b : S) -> Self
        where S : Into<DMatrix<f64>>
    {
        let a : DMatrix<f64> = a.into();
        let b : DMatrix<f64> = b.into();
        assert!(a.ncols() == b.ncols());
        let mut dst = DMatrix::zeros(a.nrows(), b.ncols());
        for (i, row_a) in a.row_iter().enumerate() {
            for (j, row_b) in b.row_iter().enumerate() {
                dst[(i, j)] = row_a.iter()
                    .zip(row_b.iter())
                    .fold(0.0, |sum, p| sum + (p.0.powf(2.) - p.1.powf(2.)).abs() as f64)
                    .sqrt();
            }
        }
        Self{ dst, dim : a.ncols() }
    }

    fn full(&self) -> DMatrix<f64> {
        self.dst.clone()
    }

    fn closest_to(&self, pt : &[f64], n : usize) -> CsMatrix<f64> {
        unimplemented!()
    }

}


