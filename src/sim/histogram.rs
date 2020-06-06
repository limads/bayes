use nalgebra::*;
use nalgebra::storage::*;
use std::cmp::Ordering;

/// One-dimensional histogram, useful for representing univariate marginal distributions
/// of sampled posteriors non-parametrically. Retrieved by indexing a Sample structure.
pub struct Histogram {

    ord_sample : DVector<f64>,

    // homogeneous partition of the variable domain.
    prob_intv : f64,

    // Values of the variable that partition the
    // cumulative sum into homogeneous probability
    // intervals.
    bounds : DVector<f64>,

    mean : f64,

    variance : f64,

    full_interval : f64
}

impl Histogram {

    pub fn build<S>(sample : &Matrix<f64, Dynamic, U1, S>) -> Self
        where S : Storage<f64, Dynamic, U1>
    {
        println!("sample = {}", sample);
        assert!(sample.nrows() > 5);
        let mut sample_c = sample.clone_owned();
        let mut sample_vec : Vec<_> = sample_c.data.into();
        sample_vec.sort_unstable_by(|s1, s2| s1.partial_cmp(s2).unwrap_or(Ordering::Equal) );
        let ord_sample = DVector::from_vec(sample_vec);
        let n = ord_sample.nrows();
        let s_min : f64 = ord_sample[0];
        let s_max : f64 = ord_sample[n - 1];
        let n_bins = n / 4;
        let mut acc = s_min + s_max;
        let mut acc_sq = s_min.powf(2.) + s_max.powf(2.);
        let mut bounds = DVector::zeros(n_bins+1);
        bounds[0] = s_min;
        bounds[n_bins] = s_max;
        let mut curr_bin = 0;
        let prob_intv = 1. / n_bins as f64;
        for i in 1..(ord_sample.nrows() - 1) {
            acc += ord_sample[i];
            acc_sq += ord_sample[i].powf(2.);
            println!("sample:{}", ord_sample[i]);
            println!("acc: {}", acc);
            if i as f64 / n as f64 > (curr_bin as f64)*prob_intv {
                bounds[curr_bin] = (ord_sample[i-1] + ord_sample[i]) / 2.;
                curr_bin += 1;
            }
        }
        assert!(curr_bin == bounds.nrows() - 1);
        let mean = acc / (n as f64);
        let variance = acc_sq / (n as f64) - mean.powf(2.);
        let full_interval = bounds[bounds.nrows() - 1] - bounds[0];
        Self{ ord_sample, prob_intv, bounds, mean, variance, full_interval }
    }

    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn var(&self) -> f64 {
        self.variance
    }

    /// Returns value such that cumulative probability == prob.
    pub fn quantile(&self, prob : f64) -> f64 {
        assert!(prob >= 0.0 && prob <= 1.0, "Invalid probability value");
        let p_ix = ((self.bounds.nrows() as f64) * prob) as usize;
        (self.bounds[p_ix] + self.bounds[p_ix - 1]) / 2.
    }

    /// Returns cumulative probability up to the informed value.
    pub fn prob(&self, value : f64) -> f64 {
        assert!(value < self.bounds[self.bounds.nrows() - 1], "Value should be < sample maximum");
        let b_ix = self.bounds.iter().position(|b| *b > value).unwrap();
        //(self.bounds[b_ix] - self.bounds[0]) / self.full_interval
        (b_ix as f64) / self.bounds.nrows() as f64
    }

    /// Returns (bin center, probabilities) pair when the full interval is
    /// partitioned into nbin intervals.
    pub fn full(&self, nbins : usize, cumul : bool) -> (DVector<f64>, DVector<f64>) {
        let interval = self.full_interval / nbins as f64;
        let mut values = DVector::from_iterator(
            nbins,
            (0..(nbins)).map(|b| self.bounds[0] + (b as f64)*interval + interval/2.)
        );
        let mut prob = values.map(|v| self.prob(v));
        if !cumul {
            for i in 1..prob.nrows() {
                prob[i] -= prob.iter().take(i-1).fold(0.0, |ps, p| ps + p);
            }
        }
        (values, prob)
    }

}


