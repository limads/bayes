use crate::prob::*;
use special::*;

#[derive(Clone)]
pub struct Beta {

}

/*
impl ExponentialFamily<Dynamic> for Beta {

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        //println!("y={}", y);
        if y.ncols() > 2 {
            panic!("The Beta distribution can only be evaluated at a single data point");
        }
        let theta = y[0];
        let theta_trunc = if theta == 0.0 {
            1E-10
        } else {
            if theta == 1. {
                1. - 1E-10
            } else {
                theta
            }
        };
        DVector::from_element(1, 1. / (theta_trunc * (1. - theta_trunc)) )
    }

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1, "Beta should be evaluated against a single column sample");
        let mut suf = DMatrix::zeros(2, 1);
        for y in y.column(0).iter() {
            let y_trunc = if *y == 0.0 {
                1E-10
            } else {
                if *y == 1. {
                    1. - 1E-10
                } else {
                    *y
                }
            };
            suf[(0,0)] += y_trunc.ln();
            suf[(1,0)] += (1. - y_trunc).ln()
        }
        suf
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(t.ncols() == 1 && t.nrows() == 2, "Sufficient probability matrix of beta should be 2x1");
        assert!(self.log_part.nrows() == 1, "Sufficient probability matrix of beta should be 2x1");
        self.ab.dot(&t.column(0)) - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self /*, eta : DVectorSlice<'_, f64>*/ ) {
        //println!("{}", eta);
        let log_part_val = Gamma::ln_gamma(self.ab[0] as f64) +
            Gamma::ln_gamma(self.ab[1] as f64) -
            Gamma::ln_gamma(self.ab[0] as f64 + self.ab[1] as f64 );
        self.log_part = DVector::from_element(1, log_part_val);
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
        DVector::from_iterator(eta.nrows(), eta.iter().map(|t| *t))
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
       DVector::from_iterator(theta.nrows(), theta.iter().map(|t| *t))
    }
}

    fn sample(&self, dst : &mut [f64]) {
        use rand_distr::Distribution;
        for i in 0..dst.len() {
            let b = self.sampler.sample(&mut rand::thread_rng());
            dst[i] = b;
        }
    }

    fn sample_into(&self, mut dst : DMatrixSliceMut<'_,f64>) {
        use rand_distr::Distribution;
        let b = self.sampler.sample(&mut rand::thread_rng());
        dst[(0,0)] = b;
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.mean
    }

    fn mode(&self) -> DVector<f64> {
        let (a, b) = (self.ab[0], self.ab[1]);
        DVector::from_column_slice(&[a - 1., a + b - 2.])
    }

    fn var(&self) -> DVector<f64> {
        let (a, b) = (self.ab[0], self.ab[1]);
        DVector::from_element(1, a*b / (a + b).powf(2.) * (a + b + 1.))
    }
*/

// Reference: https://github.com/kthohr/stats/blob/master/include/stats_incl/dens/dbeta.ipp
fn beta_log_prob(x : f64, a : f64, b : f64) -> f64 {
    -1.0*(a.ln_gamma().0 + b.ln_gamma().0 - (a + b).ln_gamma().0 ) +
        (a - 1.0)*x.ln() + (b - 1.0)*(1.0 - x).ln()
}

impl Beta {

}

impl Univariate for Beta {

}

/*impl Prior for Beta {

    //fn prior(param : &[f64]) -> (DAG<Self>, NodeIndex) {
    //    Self { }
    //}

    fn as_parent<B>(self) -> Factor<B> {
        Factor::<B>::UParent(UFactor::Beta(self))
    }

}*/
