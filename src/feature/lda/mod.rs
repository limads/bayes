use nalgebra::*;

/// LDA projects a data set into an axis of maximum between-class
/// variation (the axis perpendicular to the best separable vector),
/// to that a decision over k classes can be made with k variables
/// using information from an arbitrarily large number of dimensions,
/// or the subset of variables that contribute most to this axis
/// can be selected.
/// This method re-expresses a data set around a discriminant axis
/// based on a classification vector or design matrix.
/// The coefficients of LDA are the eigenvalues of sigma_w^-1 sigma_b;
/// The eigenvectors are the ordered, unscaled axes of maximum discriminability.
/// This structure preserve within/between-class variance (LDA).
pub struct LDA {

}

impl LDA {

    pub fn new(sample : DVector<f64>, classes : DVector<f64>) -> Self {
        unimplemented!()
    }
}
