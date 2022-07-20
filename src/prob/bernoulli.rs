use crate::prob::*;
use petgraph::adj::NodeIndex;

// #[doc = include_str!("../../docs/html/binomial.html")]

// num_integer::IterBinomial
// num_integer::multinomial

    /*fn base_measure(y : DMatrixSlice<'_, f64>) -> DVector<f64> {
        DVector::from_element(y.nrows(), 1.)
    }

    fn sufficient_stat(y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        assert!(y.ncols() == 1);
        DMatrix::from_element(1, 1, y.sum() / (y.nrows() as f64) )
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
            .for_each(|(l,e)| { *l = (1. + e.exp()).ln(); } );
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
        eta.map(|e| e.sigmoid() )
    }

    fn link<S>(theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
    where S : Storage<f64, Dynamic, U1>
    {
        theta.map(|p| p.logit() )
    }

    fn sample(&self, dst : &mut [f64]) {
        use rand_distr::{Distribution};
        let opt_theta = sample_canonical_factor::<Bernoulli,_>(self.view_fixed_values(), &self.factor, self.n);
        let theta = opt_theta.as_ref().unwrap_or(&self.theta);
        for (i, _) in theta.iter().enumerate() {
            dst[i] = (self.sampler[i].sample(&mut rand::thread_rng()) as i32) as f64;
        }
    }

    fn sample_into(&self, mut dst : DMatrixSliceMut<'_,f64>) {
        use rand_distr::{Distribution};
        let opt_theta = sample_canonical_factor::<Bernoulli,_>(self.view_fixed_values(), &self.factor, self.n);
        let theta = opt_theta.as_ref().unwrap_or(&self.theta);
        for (i, _) in theta.iter().enumerate() {
            dst[(i,0)] = (self.sampler[i].sample(&mut rand::thread_rng()) as i32) as f64;
        }
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        &self.theta
    }

    fn mode(&self) -> DVector<f64> {
        self.theta.clone()
    }

    fn var(&self) -> DVector<f64> {
        self.theta.component_mul(&self.theta.map(|p| 1. - p))
    }

    impl<'a> Estimator<'a, &'a Beta> for Bernoulli {

    type Algorithm = ();

    type Error = &'static str;

    //fn predict<'a>(&'a self, cond : Option<&'a Sample/*<'a>*/>) -> Box<dyn Sample /*<'a>*/> {
    //    unimplemented!()
    //}

    /*fn take_posterior(self) -> Option<Beta> {
        unimplemented!()
    }

    fn view_posterior<'a>(&'a self) -> Option<&'a Beta> {
        unimplemented!()
    }*/

    fn fit(&'a mut self, algorithm : Option<()>) -> Result<&'a Beta, &'static str> {
        // self.observe_sample(sample);
        // assert!(y.ncols() == 1);
        // assert!(x.is_none());
        match self.factor {
            BernoulliFactor::Conjugate(ref mut beta) => {
                let y = self.obs.clone().unwrap();
                let n = y.nrows() as f64;
                let ys = y.column(0).sum();
                let (a, b) = (beta.view_parameter(false)[0], beta.view_parameter(false)[1]);
                let new_param = DVector::from_column_slice(&[a + ys, b + n - ys]);
                beta.set_parameter(new_param.rows(0, new_param.nrows()), false);
                Ok(&(*beta))
            },
            _ => Err("Distribution does not have a conjugate factor")
        }
    }

}

*/

fn bernoulli_log_prob(x : f64, theta : f64) -> f64 {
    if x == 0.0 {
        theta.ln()
    } else {
        (1. - theta).ln()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Bernoulli {
    loc : f64
}

impl Univariate for Bernoulli {

}

/*impl Likelihood for Bernoulli {

    fn likelihood(data : &[f64]) -> Joint<Bernoulli> {
        Joint::<Bernoulli>::from_slice(data)
    }

}*/

/*impl Conditional<Beta> for Bernoulli {

    fn condition(mut self, parent : Beta, edges : &[&[f64]]) -> (DAG<Bernoulli>, NodeIndex, NodeIndex) {
        super::condition_empty(self, parent, edges[0])
    }

}

impl Conditional<Beta> for Joint<Bernoulli> {

    fn condition<'a>(mut self, parent : Beta, edges : &[&[f64]]) -> (DAG<Joint<Bernoulli>>, NodeIndex, NodeIndex) {
        super::condition_empty(self, parent, edges[0])
    }

}*/

