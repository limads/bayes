use bayes::distr::*;

fn main() -> Result<(), &'static str> {
    println!("Normal model:");
    let mut m = Normal::new(1,None,None)
        .condition(Normal::new(1,None,None))
        .condition(Gamma::new(1.,1.));
    m.visit_factors(|f| println!("Factor: {}", f) );

    println!("Bernoulli model:");
    let mut b = Bernoulli::new(1,None)
        .condition(Beta::new(1, 1));
    b.visit_factors(|f| println!("Factor: {}", f) );

    println!("Multilevel normal model:");
    let mut m1 = Normal::new(1,None,None)
        .condition(Normal::new(1,None,None));
    let mut m2 = Normal::new(1,None,None).condition(m1)
        .condition(Gamma::new(1.,1.));
    m2.visit_factors(|f| println!("Factor: {}", f) );

    println!("Poisson model:");
    let mut p = Poisson::new(1,None)
        .condition(Gamma::new(1., 1.));
    p.visit_factors(|f| println!("Factor: {}", f) );
    Ok(())
}




