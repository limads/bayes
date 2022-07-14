use crate::prob::*;
use special::*;

pub struct Beta {

}

// Reference: https://github.com/kthohr/stats/blob/master/include/stats_incl/dens/dbeta.ipp
fn beta_log_prob(x : f64, a : f64, b : f64) -> f64 {
    -1.0*(a.ln_gamma().0 + b.ln_gamma().0 - (a + b).ln_gamma().0 ) +
        (a - 1.0)*x.ln() + (b - 1.0)*(1.0 - x).ln()
}

impl Beta {

}

impl Univariate for Beta {

}

/*impl Prior for Beta {

    //fn prior(param : &[f64]) -> (DAG<Self>, NodeIndex) {
    //    Self { }
    //}

    fn as_parent<B>(self) -> Factor<B> {
        Factor::<B>::UParent(UFactor::Beta(self))
    }

}*/
