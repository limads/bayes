use bayes::prob::Normal;
use rand::thread_rng;
use bayes::prob::Graph;

// pub struct Marginal {
//    factors : Vec<Distr>
// }

fn main() {
    let mut rng = thread_rng();
    let n1 = Normal::new(0., 1.0);
    let n2 = Normal::new(10., 1.);
    let y : Vec<_> = (0..100).map(|n1| n1.sample(&mut rng))
        .chain((0..100).map(|n2| n2.sample(&mut rng) )).collect();
    let graph = Graph::new();
    let n0 = graph.add(Normal::likelihood(y.clone()));    
    let n1 = graph.add(Normal::likelihood(y));
    
    // Our likelihood distribution now is the cartesian product (y1,z=0;y1,z=1), (y2,z=0;y2,z=1)
    // and so on.
    let marginal = graph.marginalize(&[n0, n1]);
    
    // This could also be a series of Categorical(1),Categorical(2),Categorical(3)
    // and a Dirichlet prior to hold the probabilities.
    let z0 = graph.add(Bernoulli::likelihood(0));
    let z1 = graph.add(Bernoulli::likelihood(1));
    let theta = graph.add(Beta::new(0.5));
    
    // We now have two conditional posteriors:
    // theta,mu1|z=1, theta,mu2|z=2. At each MCMC iteration, evaluate both of
    // them and sum the resulting log-probabilities.
    graph.condition(z0, theta);
    graph.condition(z1, theta);
    graph.condition(n0, z0);
    graph.condition(n1, z1);
    
    match RWMH::new(&mut graph).posterior() {
        Ok(res) => {
            println!("Fit successful. R^2 = {}", res.r2);
            println!("theta = {}", graph[theta].prob());
            println!("mu0 = {}", graph[n0].mean());
            println!("mu1 = {}", graph[n1].mean());
        },
        Err(e) => {
            println!("Fit error: {}", e);
        }
    }
    
}




