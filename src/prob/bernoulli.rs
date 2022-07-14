use crate::prob::*;
use petgraph::adj::NodeIndex;

// #[doc = include_str!("../../docs/html/binomial.html")]

// num_integer::IterBinomial
// num_integer::multinomial

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

