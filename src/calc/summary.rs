use crate::prob::*;

fn mean<D>(distr : D) -> Vec<f64>
where
    D : Distribution
{
    distr.natural_parameter().data.into()
}

// Square root of the variance of observations
fn standard_error(distr : D) -> Vec<f64>
where
    D : Distribution + Likelihood
{
    let data = distr.observations().variance();
    distr.natural_parameter().data.into()
}


