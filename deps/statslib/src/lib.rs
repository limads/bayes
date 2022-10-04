fn log_if(x : f64, log_form : bool) -> f64 {
    if log_form { 
        x.ln() 
    } else { 
        x 
    }
}

fn log_zero_if(log_form : bool) -> f64 {
    if log_form {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

fn log_one_if(log_form : bool) -> f64 {
    if log_form {
        0.0
    } else {
        1.0
    }
}

fn exp_if(x : f64, exp_form : bool) -> f64 {
    if exp_form { 
        x.exp() 
    } else { 
        x 
    }
}

fn any_posinf(a : f64, b : f64) -> bool {
    a == f64::INFINITY || b == f64::INFINITY
}

fn all_posinf(a : f64, b : f64) -> bool {
    a == f64::INFINITY && b == f64::INFINITY
}

fn all_neginf(a : f64, b : f64) -> bool {
    a == f64::NEG_INFINITY && b == f64::NEG_INFINITY
}

fn is_posinf(a : f64) -> bool {
    a == f64::INFINITY
}

fn any_nan(a : f64, b : f64) -> bool {
    a.is_nan() || b.is_nan()
}

pub mod dens {

    use nalgebra::*;
    use special::*;
    
    pub fn dbern(x : u32, theta : f64, log_form : bool) -> f64 {
        let p = if x == 1 {
            theta
        } else {
            (1. - theta)
        };
        super::log_if(p, log_form)
    }
    
    fn dbeta_log_compute(x : f64, a : f64, b : f64) -> f64 {
        -1.0*(a.ln_gamma().0 + b.ln_gamma().0 - (a + b).ln_gamma().0 ) +
            (a - 1.0)*x.ln() + (b - 1.0)*(1.0 - x).ln()
    }
    
    fn dbeta_limit_vals(x : f64, a : f64, b : f64) -> f64 {
        if a == 0.0 && b == 0.0 {
            if x == 0.0 || x == 1.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else if a == 0.0 || (is_posinf(b) && !is_posinf(a)) {
            if x == 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else if b == 0.0 || (is_posinf(a) && !is_posinf(b)) {
            if x == 1.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            if x == 0.5 {
                f64::INFINITY
            } else {
                0.0
            }
        }
    }
    
    pub fn dbeta(x : f64, a : f64, b : f64, log_form : bool) -> f64 {
        if !beta_sanity_check(x, a, b) {
            return f64::NAN;
        }
        if x < 0.0 || x > 1.0 {
            return log_zero_if(log_form);
        }
        if a == 0.0 || b == 0.0 || any_posinf(a, b) {
            log_if(dbeta_limit_vals(x, a, b), log_form)
        } else {
            if x == 0.0 {
                if a < 1.0 {
                    f64::INFINITY
                } else if a > 1.0 {
                    log_zero_if(log_form)
                } else {
                    log_if(b, log_form)
                }
            } else if x == 1.0 {
                if b < 1.0 {
                    f64::INFINITY
                } else {
                    if b > 1.0 {
                        super::log_zero_if(log_form)
                    } else {
                        super::log_if(a, log_form)
                    }
                }
            } else {
                super::exp_if(dbeta_log_compute(x, a, b), !log_form)
            }
        }
    }

    fn binom_sanity_check(ntrials : u32, prob : f64) -> bool {
        if prob.is_nan() {
            false
        } else {
            if prob.is_inf() {
                false
            } else {
                if prob < 0.0 || prob > 1.0 {
                    false
                } else {
                    true
                }
            }
        }
    }
    
    fn log_binomial_coef(n : u32, k : u32) -> f64 {
        ((n+1) as f64).ln_gamma().0 - ( ((k+1) as f64).ln_gamma().0 + ((n-k+1) as f64).ln_gamma().0 )
    }
    
    fn dbinom_log_compute(x : u32, ntrials : u32, prob : f64) {
        if x == 0 {
            ntrials as f64 * (1. - prob).ln()
        } else if x == ntrials {
            x as f64 * prob.ln()
        } else {
            log_binomial_coef(ntrials, x) + 
                x as f64 * prob.ln() + 
                (ntrials - x) as f64 * (1. - prob).ln()
        }
    }
    
    fn dbinom_limit_vals(x : u32) -> u32 {
        if x == 0 {
            1
        } else {
            0
        }
    }
    
    pub fn dbinom(x : u32, ntrials : u32, prob : f64, log_form : bool) -> f64 {
        if !binom_sanity_check(ntrials, prob) {
            f64::NAN
        } else {
            if x > ntrials {
                log_zero_if(log_form)
            } else {
                if ntrials == 0 {
                    super::log_if(dbinom_limit_vals(x), log_form)
                } else if ntrials == 1 {
                    dbern(x as f64, prob, log_form)
                } else {
                    super::exp_if(dbinom_log_compute(x, ntrials, prob), !log_form)
                }
            }
        }
    }
    
    fn dexp_log_compute(x : f64, rate : f64) -> f64 {
        rate.ln() - rate * x
    }
    
    pub fn dexp(x : f64, rate : f64) -> f64 {
        if !exp_sanity_check(x, rate) {
            f64::NAN
        } else {
            if is_posinf(rate) {
                f64::NAN
            } else {
                if x < 0.0 {
                    log_zero_if(log_form)
                } else {
                    if is_posinf(x) {
                        if rate > 0. {
                            0.
                        } else {
                            f64::NAN
                        }
                    } else {
                        super::exp_if(dexp_log_compute(x, rate), !log_form)
                    }
                }
            }
        }
    }
    
    fn gamma_sanity_check(shape : f64, scale : f64) -> bool {
        if any_nan(shape, scale) {
            false
        } else {
            if shape < 0. {
                false
            } else if f64::EPS > scale {
                false
            } else {
                true
            }
        }
    }
    
    // Reference: stats::dgamma.ipp
    fn dgamma_log_compute(x : f64, shape : f64, scale : f64) -> f64 {
        -1.0*shape.ln() - shape*scale.ln() + (shape - 1.0)*x.ln() - x / scale
    }
    
    fn dgamma_limit_vals(x : f64, shape : f64, scale : f64) -> f64 {
        if shape == 0. {
            if x == 0. {
                f64::INFINITY
            } else {
                0.
            }
        } else if shape < 1.0 {
            f64::INFINITY
        } else {
            if shape == 1.0 {
                1.0 / scale
            } else {
                0.
            }
        }
    }
    
    pub fn dgamma(x : f64, shape : f64, scale : f64, log_form : bool) -> f64 {
        if !gamma_sanity_check(x, shape, scale) {
            f64::NAN
        } else {
            if x < 0. {
                super::log_zero_if(log_form)
            } else if x == 0. || shape == 0. {
                super::log_if(dgamma_limit_vals(x, shape, scale), log_form)
            } else {
                if any_posinf(x, shape, scale) {
                    super::log_zero_if(log_form)
                } else {
                    super::exp_if(dgamma_log_compute(x, shape, scale), !log_form)
                }
            }
        }
    }
    
    fn norm_sanity_check(x : f64, mu : f64, stddev : f64) -> bool {
        let param_check = if super::any_nan(mu, stddev) {
            false
        } else if stddev < 0.0 {
            false
        } else {
            true
        };
        !x.is_nan() && param_check
    }
    
    // based on stats::dnorm.ipp
    fn dnorm_log_compute(x : f64, mu : f64, stddev : f64) -> f64 {
        dnorm_log_compute_std((x - mu) / stddev, stddev)
    }

    // based on stats::dnorm.ipp
    fn dnorm_log_compute_std(z : f64, stddev : f64) -> f64 {
        -0.5 * LN_2_PI - stddev.ln() - z.powf(2.0) / 2.0
    }

    fn dnorm_limit_vals(x : f64, mu : f64, stddev : f64) -> f64 {
        if stddev.is_posinf() {
            0.
        } else {
            if all_posinf(x, mu) || all_neginf(x, mu) {
                f64::NAN
            } else {
                if stddev == 0. && x == mu {
                    f64::INFINITY
                } else {
                    0.
                }
            }
        }
    }
    
    pub fn dnorm(x : f64, mu : f64, stddev : f64, log_form : bool) -> f64 {
        if !norm_sanity_check(x, mu, stddev) {
            f64::NAN    
        } else {
            if (any_inf(x, mu) || stddev.is_inf()) || stddev == 0. {
                log_if(dnorm_limit_vals(x, mu, stddev), log_form)
            } else {
                exp_if(dnorm_log_compute(x, mu, stddev), !log_form)
            }
        }
    }
    
    // 1 / (2*pi) (normal base measure)
    const INV_2_PI : f64 = 0.1591549430918953357688837633725143620344596457404564487476673440;

    // (2*pi).ln()
    const LN_2_PI : f64 = 1.8378770664093454835606594728112352797227949472755668256343030809;

    // based on stats::dpois.ipp
    fn dpois_log_compute(x : u32, rate : f64) -> f64 {
        x as f64 * rate.ln() - rate - ((x+1) as f64).ln_gamma().0
    }
    
    pub fn dpois(x : u32, rate : f64, log_form : bool) -> f64 {
        if !pois_sanity_check(rate) {
            f64::NAN
        } else if rate == 0. {
            if x == 0 {
                log_one_if(log_form)
            } else {
                log_zero_if(log_form)
            }
        } else if rate.is_posinf() {
            log_zero_if(log_form)
        } else {
            exp_if(dpois_log_compute(x, rate), !log_form)
        }
    }

    pub fn dmvnorm(x : &DVector<f64>, mean : &DVector<f64>, cov : &DMatrix<f64>, log_form : bool) -> f64 {

        // This term can be computed at compile time if VectorN is used. Or we might
        // keep results into a static array of f64 and just index it with x.nrows().
        let partition = -0.5 * x.nrows() as f64 * LN_2_PI;

        let xc = x.clone() - mean;

        let cov_chol = Cholesky::new(cov.clone()).unwrap();

        // x^T S^-1 x
        let mahalanobis = xc.transpose().dot(&cov_chol.solve(&xc));

        let mut ret = partition - 0.5 * (cov_chol.determinant().ln() + mahalanobis);
        
        if !log_form {
            ret = ret.exp();
            if ret.is_inf() {
                ret = f64::max_value()
            }
        }
        
        ret

    }
    
}


