use bayes::prob::*;
use bayes::fit::linear::irls;
use rand;

/*/// Add a small random perturbation to each element in the vector
pub fn jittered(iter : impl Iterator<Item=f64>, ampl : f64) -> impl Iterator<Item=f64> {
    iter.map(|val| {
        let mut jitter : f64 = rand::random();
        jitter *= ampl;
        val + jitter
    })
}*/

fn main() -> Result<(), String> {
    let n = 100;
    let ys : Vec<_> = (0..100).map(|y| if y < 50 { true } else { false } ).collect();
    let mut b = Bernoulli::likelihood(ys.iter());
    println!("{:?}", b.observations());

    let xs : Vec<_> = (0..100).map(|x| {
        let mut jitter : f64 = rand::random();
        jitter *= 0.001;
        if x < 50 { [1.0, 5.0+jitter] } else { [1.0, 0.0+jitter] }
    }).collect();
    // let prior = MultiNormal::prior(&[0.0, 0.0]);
    let coefs = [1.0, 1.0];
    let m = MultiNormal::fixed(xs.iter().map(|x| &x[..] ), coefs.iter());
    println!("{:?}", m.fixed_observations());

    let mut b = b.condition(m);

    irls(&mut b, 0.0001, 1000)?;
    let mn : &MultiNormal = b.view_factor().unwrap();
    println!("{:?}", mn.view_parameter(true) );

    Ok(())
}
