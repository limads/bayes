use bayes::distr::*;

fn main() -> Result<(), &'static str> {
    let mut m = Normal::new(1,None,None).condition(Normal::new(1,None,None))
        .condition(Gamma::new(1.,1.));
    //println!("Factors: {:?}", m.factors_mut());
    m.factors_mut().visit::<_,()>(|f, _| println!("Factor: {}", f), None)?;
    Ok(())
}




