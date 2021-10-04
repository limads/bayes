use bayes::prob::Normal;
use std::collections::HashMap;
// use bayes::sample::Sample;
use bayes::prob::MultiNormal;

const N : usize = 100;

const BETA : [f64; 2] = [1.0, 2.0];

/*fn generate_observations() -> HashMap<String, Vec<f64>> {
    let data_range = 0..N;
    let err_distr = Normal::new(N, 0.0, 2.0);
    let err = err_distr.sample();
    let x1 : Vec<_> = data_range.map(|i| i as f64 / N as f64 ).collect();
    let x2 : Vec<_> = data_range.map(|i| i as f64 / N as f64 ).collect();
    let y : Vec<_> = data_range
        .map(|i| x1[i]*BETA[0] + x2[i]*BETA[1] + err[i] )
        .collect();
    let mut data = HashMap::new();
    data.insert("x1", x1);
    data.insert("x2", x2);
    data.insert("y", y);
    data
}*/

fn main() {
    /*let x = MultiNormal::new_standard(100)
        .fix(&["x1", "x2"]);
    let model = Normal::new(100, None, None)
        .observe(&["y"])
        .condition(x);
        
    let obs = generate_observations();*/
}
