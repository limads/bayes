use bayes::prob::*;
use rand::thread_rng;

fn main() -> Result<(), Box<dyn Error>> {
    let prior = Normal::prior(0.0, 0.1);
    let prior_samples : Vec<_> = (0..10).map(|_| prior.sample(thread_rng()) );
    let mut likelihood = Normal::likelihood([1., 2., 1., 2.]);
    likelihood.condition(prior);
    let posterior = likelihood.fit()?;
    let posterior_samples : Vec<_> = (0..10).map(|_| posterior.sample(thread_rng()) );
    println!("{}", posterior);
    Ok(())
}


