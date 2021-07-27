use bayes::prob::*;

fn generate() -> [f64; 200] {
    let n1 = Normal::new(Some(0.0), Some(1.0));
    let n2 = Normal::new(Some(10.0), Some(0.1));
    let data : [f64; 200] = [0.0; 200];
    n1.sample(&mut data[0..100]);
    n2.sample(&mut data[100..]);
    data
}

fn main() {
    let data = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0].iter().map(|v| v + ra)
    let n = Normal::likelihood(&);
}
