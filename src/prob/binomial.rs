use special::*;

//Reference: gcem::log_binomial_coef
fn log_binomial_coef(n : u32, k : u32) -> f64 {
    ((n+1) as f64).ln_gamma().0 - ( ((k+1) as f64).ln_gamma().0 + ((n-k+1) as f64).ln_gamma().0 )
}

// Reference: stats::dbinom.ipp
fn binomial_log_prob(x : u32, n : u32, theta : f64)->f64 {
    if x == 0 {
        (n as f64)*(1. - theta).ln()
    } else {
        if x == n {
            (x as f64) * theta.ln()
        } else {
            log_binomial_coef(n, x) + (x as f64)*theta.ln() + ((n - x) as f64)*(1. - theta).ln()
        }
    }
}

