use super::MultiNormal;

/// Semi-parametric representation of an univariate distribution. Quantiles are multinormals
/// which represent fixed user-defined cumulative probability density positions (e.g. 0.25, 0.5, 0.75), 
/// and are estimated from an empirical distribution function. Quantiles can be conditioned on MultiNormal priors
/// and can be used as conditioning factors in the same positions a multinormal can. They can be useful
/// when the quantity of interest is clearly non-normal (e.g. have many modes) but there is no interest
/// in modelling those modes as fixed or random factors.
pub struct Quantile {
    vals : MultiNormal,
    pos : Vec<usize>
}   
