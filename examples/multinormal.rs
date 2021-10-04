use bayes::prob::*;
use nalgebra::DMatrix;

/* Check against:

import numpy as np
from scipy import stats

mu = np.array([2.066666666666667,  2.3666666666666667])
cov = np.array([
	[5.626666666666667, -0.6733333333333331],
	[-0.6733333333333331, 0.6066666666666667]
])

data = [[1., 2.], [1.2, 3.0], [4.0, 2.1]]

lp = 0.0
for d in data:
	lp += stats.multivariate_normal.logpdf(d, mu, cov)

print(lp) */

fn main() {
    // Will the data points used to fit the mvn MLE all have the same mahalanobis distance
    // calculated from the MLE estimates (1/full_dist)? where full_dist is given by the log-
    // partition, which will be -1 if the log-partition is ignored?
    /*let data : [[f64; 2]; 3] = [[1.0, 2.0], [1.2, 3.0], [4.0, 2.1]];
    let mut m = MultiNormal::likelihood(data.iter().map(|d| &d[..] ));
    print!("{}", m.observations().as_ref().unwrap());
    print!("{}\n{}\n", m.view_parameter(false), m.cov().unwrap());
    let cond = m.cond_log_prob().unwrap();
    println!("cond log prob = {}", cond);

    // let y = DMatrix::zeros(2, 1);

    // let bm = MultiNormal::base_measure((&y).into())[0];
    // println!("joint log prob = {}", 3.*bm.ln() + m.joint_log_prob().unwrap() );

    println!("summed cond log prob = {}", cond.sum());*/
}


