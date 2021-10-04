// runner examples/regression.rs -s

extern crate nalgebra;
extern crate bayes;
extern crate rand_distr;
extern crate rand;

use bayes::prob::{Normal, MultiNormal};
use bayes::fit::Fixed;
use std::default::Default;
use rand_distr::Distribution;

fn multivariate_regression() {
    let y = [[1.0, 2.0, 1.1, 2.0], [1.0, 2.0, 1.1, 2.0]];
    let x = [[1.0, 0.0, 1.1, 1.0], [1.0, 2.0, 1.1, 2.0]];
    let y : [Normal; 2] = [Normal::likelihood(y[0]), Normal::likelihood(y[1])];
    y.condition([MultiNormal::fixed(x.clone()), MultiNormal::fixed(x)]);
}

fn main() -> Result<(), String> {
    let y = [1.0, 2.0, 1.1, 2.0];
    let x = [[1.0, 0.0, 1.1, 1.0], [1.0, 2.0, 1.1, 2.0]];

    let n : [Normal; 5] = Normal::likelihood([y]);

    // WLS --
    n[0].set_scale(1.0);
    n[1].set_scale(2.0);
    n[3].set_scale(0.0);

    // "fixed" means: Cache the observations to calculate conditional expectations, instead of
    // calculating the MLE.
    n.condition(MultiNormal::fixed([x1, x2]));
    n.fit();

    // BLS.
    n.condition(MultiNormal::fixed([x1, x2]).condition(MultiNormal::prior(&[0.0, 0.1]));
    n.fit();

    // Advantages: Peserve Exponential api of location(.), scale(.) and sample(.) for each distribution
    // individually.

    // GLM:
    let b = [Bernoulli; 3] = Bernoulli::joint([0.0, 0.1, 0.2]);
    b.condition(MultiNormal::fixed([[0., 1., 2.], [1., 2., 3]]));
    let w = b.factor_mut().coefficients();

    // Option 2:
    let y = Normal::likelihood(y)
        .fixed([[1., 2.], [3., 4.]])
        .condition(MultiNormal::prior([2.0, 1.0]));

    let y = [0.0, 1.0, 2.0].iter().zip([[1.0], [2.0]])  // impl Iterator<Item=(f64, [f64; 2])>
        .map(|y, x| Normal::likelihood(y).fixed(x) )    // impl Iterator<Item=Normal>
        .collect::<Box<[Normal]>>()                     // Box<[Normal]>
        .condition(MultiNormal::prior(0.2) )
        .fit();

    // Let m = MultiNormal::posterior(OLS::estimate(y, OLSSettings { fixed : 1.. }))
    Ok(())
}

