use crate::gsl::bpsline::*;

/// The basis splines expand an 1-dimensional domain (usually representing
/// a continuous independent variable) to contain the values of a local
/// polynomial expanded up to degree k. The basis implementation returns
/// an empty coefficient iterator, because usually they will be estimated
/// rather than be a part of the expansion.
pub struct Spline {
    ws : *mut gsl_bspline_workspace,
    basis : DMatrix<f64>
}

impl Spline {

    pub fn new(data : &DVector<f64>, breaks : usize) -> Self {
        let n = nrows(data);
        let a = d.min();
        let b = d.max();
        unsafe {
            let ws : gsl_bspline_workspace = gsl_bspline_alloc(4, breaks);
            let res = gsl_bspline_knots_uniform(a, b, ws);
            let knots = ws.knots;
            let n_coefs = gsl_bspline_ncoeffs(ws);
            let mut basis = DMatrix::zeros(n, n_coefs);
            let mut v = gsl_vector_alloc(n_coefs);
            let mut s = slice::from_raw_parts(v.data, n_coefs);
            for (i, d) in data.enumerate() {
                gsl_bspline_eval(d, v, ws);
                basis.column_mut(i).copy_from_slice(s);
            }
            gsl_vector_free(v);
            Self{ ws, basis }
        }
    }

}

impl Drop for Spline {

    fn drop(&mut self) {
         gsl_bspline_free(self.ws);
    }
}

impl Basis<DVector<f64>,C> for Spline {

}
