use nalgebra::*;

pub struct PCA {

}

impl PCA {

    pub fn decompose(y : DMatrix<f64>) -> Self {
        unimplemented!()
    }

    /// Select the k-first column indices that concentrate most of
    /// the variation, from the ordered factor loadings.
    pub fn first_variables(&self, k : usize) -> Vec<usize> {
        unimplemented!()
    }

    /// Select the components that account for at least p% of
    /// the variation present in the sample.
    pub fn first_components(&self, p : f64) -> DMatrixSlice<'_, f64> {
        unimplemented!()
    }

}


