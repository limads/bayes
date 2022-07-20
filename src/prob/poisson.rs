use crate::prob::*;
use special::*;

// based on stats::dpois.ipp
fn poisson_log_prob(x : u32, rate : f64) -> f64 {
    x as f64 * rate.ln() - rate - ((x+1) as f64).ln_gamma().0
}

#[derive(Clone)]
pub struct Poisson(f64);

impl Univariate for Poisson {

}

/*
impl ExponentialFamily<U1> for Poisson
    where
        Self : Distribution
{

    fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        assert!(y.ncols() == 1);
        y.column(0).map(|y| /*1. / (Bernoulli::factorial(y as usize) as f64*/ Gamma::gamma_inv(y + 1.) )
    }

    /*fn grad(&self, y : DMatrixSlice<'_, f64>, x : Option<DMatrix<f64>>) -> DVector<f64> {
        // equivalent to sum_i { yi/lambda - 1 }
        let g = y.nrows() as f64 * (y.mean() - self.lambda[0]) / self.lambda[0];
        DVector::from_element(1, g)
    }*/

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1);
        DMatrix::from_element(1, 1, y.column(0).iter().fold(0.0, |ys, y| ys + y ))
    }

    fn suf_log_prob(&self, t : DMatrixSlice<'_, f64>) -> f64 {
        assert!(t.nrows() == 1 && t.ncols() == 1);
        assert!(self.eta.nrows() == 1);
        assert!(self.log_part.nrows() == 1);
        self.eta[0] * t[0] - self.log_part[0]
    }

    fn update_log_partition<'a>(&'a mut self, /*eta : DVectorSlice<'_, f64>*/ ) {
        if self.log_part.nrows() != self.eta.nrows() {
            self.log_part = DVector::zeros(self.eta.nrows());
        }
        self.log_part.iter_mut()
            .zip(self.eta.iter())
            .for_each(|(l,e)| { *l = e.exp() } );
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
        eta.map(|e| e.exp() )
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        theta.map(|p| p.ln() )
    }

}

fn sample_into(&self, mut dst : DMatrixSliceMut<'_,f64>) {
        use rand_distr::{Distribution};
        let opt_lambda = sample_canonical_factor::<Poisson,_>(self.view_fixed_values(), &self.factor, self.n);
        let lambda = opt_lambda.as_ref().unwrap_or(&self.lambda);
        for (i, _) in lambda.iter().enumerate() {
            let s : u64 = self.sampler[i].sample(&mut rand::thread_rng());
            dst[(i,0)] = s as f64;
        }
    }

impl<'a> Estimator<'a, &'a Gamma> for Poisson {

    type Algorithm = ();

    type Error = &'static str;

    //fn predict<'a>(&'a self, cond : Option<&'a Sample/*<'a>*/>) -> Box<dyn Sample /*<'a>*/ > {
    //    unimplemented!()
    //}

    /*fn take_posterior(self) -> Option<Gamma> {
        unimplemented!()
    }

    fn view_posterior<'a>(&'a self) -> Option<&'a Gamma> {
        unimplemented!()
    }*/

    fn fit(&'a mut self, algorithm : Option<()>) -> Result<&'a Gamma, &'static str> {
        // self.observe_sample(sample);
        match self.factor {
            PoissonFactor::Conjugate(ref mut gamma) => {
                let y = self.obs.clone().unwrap();
                let n = y.nrows() as f64;
                let ys = y.column(0).sum();
                let (a, b) = (gamma.view_parameter(false)[0], gamma.view_parameter(false)[1]);
                let new_param = DVector::from_column_slice(&[a + ys, b + n]);
                gamma.set_parameter(new_param.rows(0, new_param.nrows()), false);
                Ok(&(*gamma))
            },
            _ => Err("Distribution does not have a conjugate factor")
        }
    }

}

*/