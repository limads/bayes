use bayes::sample::Sample;
use std::default::Default;

fn main() {
    let tbl = Sample::open("examples/table.csv", Default::default()).unwrap();
    println!("CSV representation:\n");
    println!("{}", tbl);
    println!("SQL representation:\n");
    println!("{}", tbl.create_stmt("test", true).unwrap());
    println!("Raw matrix representation:");
    println!("{}", tbl.take_data());
}


