use bayes::table::Table;

fn main() {
    let tbl = Table::open("examples/table.csv").unwrap();
    println!("CSV representation:\n");
    println!("{}", tbl);
    println!("SQL representation:\n");
    println!("{}", tbl.create_stmt("test", true).unwrap());
    println!("Raw matrix representation:");
    println!("{}", tbl.take_data());
}


