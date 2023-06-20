use num_traits::AsPrimitive;

/// Kolmogorov-Smirnov statistic: Maximum over which two Empirical CDFs
/// or an Emprirical and analytical CDF differ.
#[derive(Debug, Clone, Copy)]
pub struct KS {
    pub val : f64,
    pub diff : f64
}

#[derive(Debug, Clone)]
pub struct Empirical<D>
where
    D : AsPrimitive<f64>
{
    pub domain : Vec<D>,
    pub cprobs : Vec<f64>
}

impl<D> Empirical<D>
where
    D : AsPrimitive<f64>
{

    pub fn from_cumulative_frequencies<T>(domain : Vec<D>, cfreqs : Vec<T>) -> Self
    where
        T : AsPrimitive<f64>
    {
        let fmax : f64 = (cfreqs.last().unwrap().as_() + std::f64::EPSILON);
        assert!(fmax >= 0.0);
        let acc_probs : Vec<f64> = cfreqs.iter().map(|cf| cf.as_() / fmax ).collect();
        assert!(*acc_probs.last().unwrap() <= 1.0);
        Self::from_cumulative_probabilities(domain, acc_probs)
    }

    pub fn from_cumulative_probabilities(domain : Vec<D>, cprobs : Vec<f64>) -> Self {
        assert!(domain.len() == cprobs.len());
        Self { domain, cprobs }
    }

    pub fn empirical_ks(&self, other : &Self) -> KS {
        assert!(self.domain.len() == other.domain.len());
        let mut max_prob_diff = 0.0;
        let mut max_val = 0.0;
        for i in 0..self.domain.len() {
            let pdiff = (self.cprobs[i] - other.cprobs[i]);
            if pdiff > max_prob_diff {
                max_prob_diff = pdiff;
                max_val = self.domain[i].as_();
            }
        }
        KS { val : max_val, diff : max_prob_diff }
    }

    pub fn analytical_ks(&self, model : &impl statrs::distribution::ContinuousCDF<f64, f64>) -> KS {
        let mut max_prob_diff = 0.0;
        let mut max_val = 0.0;
        for i in 0..self.domain.len() {
            let v : f64 = self.domain[i].as_();
            let p = model.cdf(v);
            let pdiff = (p - self.cprobs[i]).abs();
            if pdiff > max_prob_diff {
                max_prob_diff = pdiff;
                max_val = v;
            }
        }
        KS { val : max_val, diff : max_prob_diff }
    }

}


