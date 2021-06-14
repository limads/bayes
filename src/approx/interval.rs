/*#[derive(Debug, Clone, Copy)]
pub enum Strategy {

    // Adds the current sample to the already-existing sample.
    Increment,

    /* Calling this creates a dynamic interval that does not grow unboundedly after each call to update(.),
    but rather forgets the first samples and updates the contained interval values circularly from the beginning, with
    a circle size defined by the current sample size.*/
    Forget
}*/

/// Trait shared by structures that represents a bounded interval in the real line.
pub trait Interval {

    fn low(&self) -> f64;

    fn high(&self) -> f64;

    /// Verify if value is within the closed interval (self.low(), self.high())
    fn contains(&self, val : &f64) -> bool {
        *val >= self.low() && *val <= self.high()
    }

    // fn update(&mut self, val : &f64, strat : Strategy);

}

/// Generic quantile interval. Represents a non-parametric distribution in terms of a low quantile
/// interal value and a high quantile interval value. Created by Distribution::quantiles(0.1) ->impl Iterator<Item=Quantile>
/// where the resulting intervals have the same probability. Distribution::median() equals
/// Distribution::quantiles(0.5); Distribution::quartile() equals distribution::quantiles(0.25).
pub struct Quantile {
    low : f64,
    high : f64
}

impl Quantile {

    /// Creates the interval by informing a low and high quantile.
    pub fn new<'a>(sample : impl Iterator<Item=&'a f64>, low : f64, high : f64) -> Self {
        unimplemented!()
    }
}

/// Represents a central interval in terms of a central mean statistic the mean squared error (variance) of the observations.
/// Constructed from a normal distribution Normal::zscore(val) where val is the desired standardized z-score corresponding
/// to the interval.
pub struct ZScore {
    low : f64,
    high : f64
}

impl ZScore {

    /// Creates the interval by informing how many standard error units away from the mean to establish the interval
    pub fn new<'a>(sample : impl Iterator<Item=&'a f64>, z : f64) -> Self {
        unimplemented!()
    }
}

/// Represents a pair of arbitrary percentile cutoff points.
pub struct Percentile {

}


/*/// Represents a central interval in terms of a median and mean absolute error of the observations.
pub struct MedInterval {

}

impl MedInterval {

    /// Creates the interval by informing how many standard absolute error units away from the median to establish the interval
    pub fn new<'a>(sample : impl Iterator<Item=&'a f64>, z : f64) -> Self {
        unimplemented!()
    }
}*/

