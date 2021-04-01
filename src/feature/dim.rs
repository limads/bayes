/// Principal Component Analysis
pub struct PCA {

}

/// Basis reductions based on decomposition of the empirical covariance matrix.
/// Those transformations project samples to the orthogonal axis that preserve
/// global variance (PCA)
impl PCA {

    /*fn components(&self) -> impl Iterator<Item=Component> {
        unimplemented!()
    }*/
}

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

    /*fn discriminants(&self) -> impl Iterator<Item=Component> {
        unimplemented!()
    }*/
    
}

/// An orthogonal principal component returned by PCA
pub struct Component {

}

/// An orthogonal discriminant axis returned by LDA
pub struct Discriminant {

}
