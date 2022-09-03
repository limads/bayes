use std::ops::Range;
use std::cmp::{PartialOrd, Ordering};

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

/// Taking a closure that establishes a possible interval around some value U, recursively calls the function
/// to establish new intervals. A single interval can be added at each iteration, and the value depends on the output of the comparator
/// function.
pub fn single_interval_search<F ,T, U>(
    vals : &[T],
    intervals : &mut Vec<(Range<usize>, U)>,
    step : usize,
    f : F,
    recursive : bool
) where
    F : Fn(&[T], Range<usize>, usize) -> Option<(Range<usize>, U)> + Clone + Copy,
    U : PartialOrd + Clone
{
    let len_before = intervals.len();
    add_disjoint_intervals(vals, intervals, step, f);
    if intervals.len() > len_before {
        let new_candidates = &intervals[len_before..];
        let best_candidate = new_candidates.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal) )
            .cloned().unwrap();
        intervals.truncate(len_before);
        intervals.push(best_candidate);
        intervals.sort_by(|a, b| a.0.start.cmp(&b.0.start) );
        if recursive {
            single_interval_search(vals, intervals, step, f);
        }
    }
}

/// Taking a closure that establishes a possible interval around some value U, recursively calls the function
/// to establish new intervals, as long as at least one valid interval is returned at the current iteration.
/// Many intervals over disjoint areas can be added at the same iteration.
pub fn multi_interval_search<F, T, U>(
    vals : &[T],
    intervals : &mut Vec<(Range<usize>, U)>,
    step : usize,
    f : F,
    recursive : bool
) where
    F : Fn(&[T], Range<usize>, usize) -> Option<(Range<usize>, U)> + Clone + Copy
{
    let len_before = intervals.len();
    add_disjoint_intervals(vals, intervals, step, f);
    let any_new = len_before < intervals.len();
    if any_new && recursive {
        multi_interval_search(vals, intervals, step, f);
    }
}

pub fn add_disjoint_intervals<F, T, U>(vals : &[T], intervals : &mut Vec<(Range<usize>, U)>, step : usize, f : F)
where
    F : Fn(&[T], Range<usize>, usize) -> Option<(Range<usize>, U)> + Clone + Copy
{
    match intervals.len() {
        0 => {
            if let Some(d1) = f(vals, Range { start : 0, end : vals.len() }, step) {
                intervals.push(d1);
            }
        },
        1 => {
            let start_range = Range { start : 0, end : intervals[0].0.start };
            if let Some(ds) = f(vals, start_range, step) {
                intervals.push(ds);
            }
            let end_range = Range { start : intervals[0].0.end, end : vals.len() };
            if let Some(de) = f(vals, end_range, step) {
                intervals.push(de);
            }
        },
        n => {

            // New intermediate intervals are always pushed to the end of the vector so as not
            // no interfere with already-inserted indices (the vector is sorted at the end
            // before the next iteration).

            let start_range = Range { start : 0, end : intervals[0].0.start };
            if let Some(ds) = f(vals, start_range, step) {
                intervals.push(ds);
            }

            for i in 0..(intervals.len()-1) {
                let middle_range = Range { start : intervals[i].0.end, end : intervals[i+1].0.start };
                if let Some(di) = f(vals, middle_range, step) {
                    intervals.push(di);
                }
            }

            if let Some(de) = f(vals, Range { start : intervals[n-1].0.end, end : vals.len() }, step) {
                intervals.push(de);
            }
        }
    }

    // Guarantee interval order after any new insertions at the end.
    intervals.sort_by(|a, b| a.0.start.cmp(&b.0.start) );
}