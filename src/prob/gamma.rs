use crate::prob::*;
use special::*;

/*
impl ExponentialFamily<Dynamic> for Gamma {

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64>
        //where S : Storage<f64, Dynamic, Dynamic>
    {
        if y.ncols() > 2 {
            panic!("The Gamma distribution can only be evaluated at a single data point");
        }
        DVector::from_element(1, 1.)
    }

    /// y: Vector of precision/rate draws
    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1, "Sample should have single column");
        let mut suf = DMatrix::zeros(2, 1);
        for y in y.column(0).iter() {
            assert!(*y > 0.0, "Gamma should be evaluated against strictly positive values.");
            suf[(0,0)] += y.ln();
            suf[(1,0)] += y;
        }
        suf
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(self.log_part.nrows() == 1, "Log-partition 1x1");
        assert!(t.ncols() == 1 && t.nrows() == 2, "Sufficient statistic matrix should be 2x1");
        self.eta.dot(&t.column(0)) - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self, /*eta : DVectorSlice<'_, f64>*/ ) {
        let log_part_v = Gamma::ln_gamma(self.eta[0] + 1.) - (self.eta[0] + 1.)*(-1.*self.eta[1]).ln();
        self.log_part = DVector::from_element(1, log_part_v);
    }

    fn log_partition<'a>(&'a self) -> &'a DVector<f64> {
        &self.log_part
    }

    /*fn update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }*/

    fn link_inverse<S>(eta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        let theta_0 = eta[0] + 1.;
        let theta_1 = (-1.)*eta[1];
        DVector::from_column_slice(&[theta_0, theta_1])
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        let eta_0 = theta[0] - 1.;
        let eta_1 = (-1.)*theta[1];
        DVector::from_column_slice(&[eta_0, eta_1])
    }
}

    fn sample(&self, dst : &mut [f64]) {
        use rand_distr::Distribution;
        for i in 0..dst.len() {
            let g = self.sampler.sample(&mut rand::thread_rng());
            dst[i] = g;
        }
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.mean
    }

    fn mode(&self) -> DVector<f64> {
        // For alpha (shape) <= 1, the mode of the beta is truncated at zero.
        // For alpha >> 1, the mode approaces the mode of the gaussian at alpha / beta.
        DVector::from_element(1, (self.ab[0] - 1.) / self.ab[1])
    }

    fn var(&self) -> DVector<f64> {
        DVector::from_element(1, self.ab[0] / self.ab[1].powf(2.))
    }

*/

// Reference: stats::dgamma.ipp
fn gamma_log_prob(x : f64, shape : f64, scale : f64) -> f64 {
    -1.0*shape.ln() - shape*scale.ln() + (shape - 1.0)*x.ln() - x / scale
}

#[derive(Clone)]
pub struct Gamma {

}

impl Univariate for Gamma {

}
