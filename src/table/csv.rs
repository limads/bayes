use ::csv;
use std::fs::File;
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use std::io::{self, Read, Write};
use nalgebra::Scalar;
use std::fmt::Display;
use std::str::FromStr;
// use std::convert::TryFrom;
use nalgebra::base::RowDVector;
// use std::boxed::Box;

/// Read from a text file, returning its contents as a String
pub fn load_content_from_file(path : &str) -> Result<String,()> {
    match File::open(path) {
        Ok(mut f) => {
            let mut content = String::new();
            f.read_to_string(&mut content)
                .map_err(|e| { println!("{}", e); () })?;
            if content.len() == 0 {
                println!("Empty file content");
                return Err(());
            }
            Ok(content)
        },
        Err(e) => {
            println!("{}", e);
            Err(())
        }
    }
}

pub fn save_content_to_file(content : &str, path : &str) -> Result<(), ()> {
    let mut f = File::open(path).map_err(|e|{ println!("{}", e); () })?;
    f.write(content.as_bytes()).map_err(|e|{ println!("{}", e); () })?;
    Ok(())
}

pub fn load_from_stdin() -> Result<String,()> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer).map_err(|_| ())?;
    Ok(buffer)
}

pub fn split_on_blank_line(content : String) -> Vec<String> {
    content.split("\n\n").map(|s| s.to_string() ).collect()
}

/// Preserve this one!
pub fn parse_header(
    csv_reader : &mut csv::Reader<&[u8]>
) -> Option<Vec<String>> {
    let mut header_entries = Vec::new();
    if let Ok(header) = csv_reader.headers() {
        for entry in header.iter() {
            let e = entry.to_string();
            header_entries.push(e);
        }
        Some(header_entries)
    } else {
        None
    }
}

/// CSV files might have unnamed columns. In this case,
/// attribute arbirtrary names "Column {i}" for i in 1..k
/// to the columns, and return them as the first tuple element.
/// Consider the first line as actual data and return them
/// in the second tuple element. If the first line has
/// valid names, return None. The csv crate considers
/// the first row as a header by default, so we should check that
/// we don't have a "pure" data file.
pub fn try_convert_header_to_data(header : &[String]) -> Option<(Vec<String>, Vec<String>)> {
    let mut new_header = Vec::new();
    let mut first_line = Vec::new();
    for (i, e) in header.iter().enumerate() {
        match e.parse::<f64>() {
            Ok(f) => {
                new_header.push(String::from("(Column ") + &i.to_string() + ")");
                first_line.push(f.to_string());
            },
            Err(_) => { }
        }
    }
    if new_header.len() == header.len() {
        Some((new_header, first_line))
    } else {
        None
    }
}

/// Preserve this one!
/// Given a textual content as CSV, return a HashMap of its columns as strings.
pub fn parse_csv_as_text_cols(
    content : &String
) -> Result<Vec<(String, Vec<String>)>, String> {
    // println!("Received content: {}", content);
    let mut csv_reader = csv::Reader::from_reader(content.as_bytes());
    let header : Vec<String> = parse_header(&mut csv_reader)
        .ok_or("No CSV header at informed file".to_string())?;
    let maybe_header_data = try_convert_header_to_data(&header[..]);
    let data_keys = match &maybe_header_data {
        Some((header, _)) => header.clone(),
        None => header.clone()
    };
    let mut data_vec : Vec<(String, Vec<String>)> = Vec::new();
    for d in data_keys.iter() {
        data_vec.push( (d.clone(), Vec::new()) );
    }
    if let Some((_,first_data_row)) = &maybe_header_data {
        for (i, (_, v)) in data_vec.iter_mut().enumerate() {
            v.push(first_data_row[i].clone());
        }
    }
    let mut n_records = 0;
    for (ix_rec, row_record) in csv_reader.records().enumerate() {
        if let Ok(row) = row_record {
            n_records += 1;
            let mut n_fields = 0;
            for (i, entry) in row.iter().enumerate() {
                if let Some((_,v)) = data_vec.get_mut(i) {
                    v.push(entry.to_string());
                    n_fields += 1;
                } else {
                    return Err("Unable to get mutable reference to data vector".into())
                }
            }
            match n_fields {
                0 => { return Err(format!("Record {:?} (Line {}) had zero fields", row, ix_rec)); },
                _ => { }
            }
        } else {
            return Err(format!("Error parsing CSV record {:?} (Line {})", row_record, ix_rec));
        }
    }
    match n_records {
        0 => Err("No records available.".to_string()),
        _ => Ok(data_vec)
    }
}

/// Preserve this one!
pub fn parse_csv_as_text_rows(content : &String) -> Option<Vec<Vec<String>>> {
    let mut rows : Vec<Vec<String>> = Vec::new();
    let mut csv_reader = csv::Reader::from_reader(content.as_bytes());
    if let Some(h) = parse_header(&mut csv_reader) {
        rows.push(h);
    }
    for row_record in csv_reader.records() {
        if let Ok(row) = row_record {
            let mut row_s = Vec::<String>::new();
            row.iter().for_each(|e|{ row_s.push(e.into()); });
            rows.push(row_s);
        }
    }
    Some(rows)
}

/// Preserve this one!
/// Try to parse the whole text table as a given DVector type,
/// preserving the names.
pub fn try_parse_col_vectors<N>(
    cols : HashMap<String, Vec<String>>
) -> Option<HashMap<String, DVector<N>>>
    where
        N : Scalar + Display + FromStr {
    let ans : HashMap<String, Option<DVector<N>>> =
        cols.iter().map(|(k, v)| (k.clone(), try_dvec_from_col::<N>(&v)) ).collect();
    if ans.values().any(|v| v.is_none()) {
        return None
    } else {
        return Some( ans.iter().map(|(k, v)| (k.clone(), v.clone().unwrap())).collect() )
    }
}

/// Give up on this one
/// Parse a CSV file as a map from column names to vectors with same dimension.
/// If any of the elements on a column cannot be parsed as float, return None
/// for the corresponding column instead of the vector.
/*pub fn parse_csv_unpacked(
    content : &String
) -> HashMap<String, Option<DVector<f32>>> {
    if let Ok(data_map) = parse_csv_as_text_cols(content) {
        let mut final_map = HashMap::new();
        for (key, col) in data_map {
            final_map.insert(key, parse_string_col(col));
        }
        final_map
    } else {
        println!("Error parsing CSV. Returning empty HashMap.");
        HashMap::new()
    }
}*/

/// Keep this one! --- Entry point for a table (column order does not matter)
/// Column loader generic over primitive type.
pub fn try_parse_col<T>(col : &Vec<String>) -> Option<Vec<T>>
    where T : FromStr
{
    let mut parsed = Vec::<T>::new();
    let mut all_parsed = true;
    for s in col.iter() {
        // println!("Found entry {}", s);
        if let Ok(d) = (*s).parse::<T>() {
            parsed.push(d);
        } else {
            // println!("could not parse some entries as numbers");
            all_parsed = false;
        }
    }
    if all_parsed {
        Some(parsed)
    } else {
        None
    }
}

pub fn try_dvec_from_col<N>(col : &Vec<String>) -> Option<DVector<N>>
    where N : Scalar + FromStr
{
    Some( DVector::<N>::from_vec(try_parse_col::<N>(col)?))
}

pub fn try_dvec_from_row<N>(row : &Vec<String>) -> Option<RowDVector<N>>
    where N : Scalar + FromStr
{
    Some( RowDVector::<N>::from_vec(try_parse_col::<N>(row)?))
}

/// Keep this one!
/// Generate a table from a matrix by mapping the given names to its column vectors,
/// preserving the order of names.
pub fn packed_into_table<N>(
    m : DMatrix<N>,
    labels : Vec<String>
) -> Option<HashMap<String, DVector<N>>>
    where
        N : Scalar + FromStr
{
    match m.ncols() == labels.len() {
        true => {
            let mut table = HashMap::<String, DVector<N>>::new();
            for (c, data) in labels.iter().zip(m.column_iter()) {
                table.insert(c.clone(), DVector::<N>::from(data));
            }
            Some(table)
        },
        false => None
    }
}

/// Give up on this one.
/// Given a vector of strings, parse each entry as f32. If all entries
/// are successfully parsed, return a vector with them. Return None otherwise.
/*fn parse_string_col(col : Vec<String>) -> Option<DVector<f32>> {
    let mut data = Vec::new();
    let mut all_parsed = true;
    for s in col.iter() {
        println!("Found entry {}", s);
        if let Ok(d) = (*s).parse::<f32>() {
            data.push(d);
        } else {
            println!("could not parse some entries as numbers");
            all_parsed = false;
        }
    }
    if all_parsed {
        Some(DVector::<f32>::from_vec(data))
    } else {
        None
    }
}*/

/// Parse some CSV text as a dynamically allocated matrix.
/// If any of the entries cannot be parsed as a floating point
/// value, return an error. Return the parsed values packed in a
/// matrix otherwise.
/// This keeps the matrix in the same order the data is organized
/// over the file. Existing headers (if any) are ignored.
pub fn load_matrix_from_string<N>(
    content : &String
) -> Result<DMatrix<N>, &'static str>
    where
        N : Scalar + FromStr
{
    let rows = parse_csv_as_text_rows(&content)
        .ok_or("Could not parse rows as text")?;
    let mut data : Vec<RowDVector<N>> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        if i == 0 {
            if let Some(h) = try_dvec_from_row::<N>(r) {
                data.push(h);
            }
        } else {
            match try_dvec_from_row::<N>(r) {
                Some(parsed_row) => {
                    data.push(parsed_row);
                },
                None => {
                    println!("Error at line {}", i);
                    return  Err("Could not parse row as numeric type");
                }
            }
            // println!("{}", i);
        }
    }
    match data.len() {
        0 => Err("No rows were parsed"),
        _ => Ok(DMatrix::<N>::from_rows(&data[..]))
    }
}

pub fn print_packed_sequence<N : Scalar + Display>(v : &DVector<N>) {
    println!("C0,C1");
    for (i, e) in v.iter().enumerate() {
        println!("{},{}", i,e);
    }
}

pub fn print_packed<N>(m : &DMatrix<N>)
    where N : Scalar + Display + FromStr
{
    println!("{}", build_string_packed(m))
}

pub fn build_string_packed<N>(m : &DMatrix<N>) -> String
    where N : Scalar + Display + FromStr
{
    let mut content = String::new();
    for r in m.row_iter() {
        let mut row_iter = r.iter();
        row_iter.next().map(|el|{ content += &format!("{}", el); });
        row_iter.for_each(|el|{ content += &format!(", {}", el); });
        content += "\n";
    }
    content
}

pub fn load_matrix_from_file<N>(path : &str) -> Option<DMatrix<N>>
    where N : Scalar + FromStr
{
    let content = load_content_from_file(path).ok()?;
    load_matrix_from_string(&content)
        .map_err(|e| println!("{}", e) )
        .ok()
}

pub fn save_matrix_to_file<N>(m : &DMatrix<N>, path : &str)
-> Result<(),()>
    where N : Scalar + FromStr + Display
{
    let content = build_string_packed(m);
    save_content_to_file(&content, path)
}

pub fn load_batch_content(paths : &str) -> Result<Vec<String>,()> {
    let paths : Vec<&str> = paths.split(",").collect();
    let opt_cont : Vec<Option<String>> = paths
        .iter().map(|p| load_content_from_file(p).ok() )
        .take_while(|o| o.is_some() ).collect();
    if opt_cont.len() == paths.len() {
        let cont : Vec<String> = opt_cont
            .iter().map(|opt_m| opt_m.clone().unwrap() ).collect();
        Ok(cont)
    } else {
        Err(())
    }
}

pub fn load_batch_packed<N>(cont : Vec<String>) -> Option<Vec<DMatrix<N>>>
    where N : Scalar + FromStr
{
    let mut mtxs : Vec<DMatrix<N>> = Vec::new();
    for c in cont {
        match load_matrix_from_string(&c) {
            Ok(m) => mtxs.push(m),
            Err(e) => { println!("{}", e); return None; }
        }
    }
    Some(mtxs)
}

/*pub fn load_batch_tables<N>(
    batch : Vec<String>
) -> Option<Vec<HashMap<String, DVector<N>>>>
where
    N : Scalar + Display
{
    let parsed_batch = Vec::new();
    for content in batch {
        let table = match parse_csv_as_text_cols(&content) {
            Ok(cols) => {
                let parsed_tbl = cols.iter().map(|(k, c)| {
                    let parsed_col = match c.try_dvec_from_col::<N>() {
                        Ok(p) => p,
                        None => { return None; }
                    };
                    (k.clone(), parsed_col)
                }).collect();
                parsed_tbl
            },
            Err(e) => { println!("{}, e"); return None; }
        };
        parsed_batch.push(table);
    }
    parsed_batch
}*/

/*mod args {

    use std::env;
    use std::collections::HashMap;

    pub fn read_io_args() -> HashMap<String, String> {
        let from_1st = env::args();
        let from_2nd = env::args().skip(1);
        from_1st.zip(from_2nd).collect()
    }

    pub fn fetch_io_optional(
        args : HashMap<String, String>
    ) -> (Option<String>, Option<String>) {
        (args.get("-i").map(|s| s.clone()),
            args.get("-o").map(|s| s.clone()))
    }

    pub fn fetch_io_required(
        args : HashMap<String, String>
    ) -> Option<(String, String)> {
        match (args.get("-i"), args.get("-o")) {
            (Some(in_file), Some(out_file)) => Some( (in_file.clone(), out_file.clone()) ),
            _ => None
        }
    }
}*/

/*mod filters {

    use nalgebra::*;
    use crate::csv::*;
    use std::fs::File;
    use std::convert::From;
    use std::fmt::Display;
    use std::str::FromStr;

    // trait DisplayableScalar = Scalar + Display + From<f32>;

    pub fn from_file_or_stdin<N>(
        path : Option<&str>
    ) ->Result<DMatrix<N>, ()>
        where
            N : Scalar + FromStr
    {
        let content = match path {
            Some(path) => load_content_from_file(path)?,
            None => load_from_stdin()?
        };
        let m_f32 = load_matrix_from_string::<N>(&content)
            .map_err(|e|{ println!("{}", e); () })?;
        let m  = DMatrix::<N>::from_iterator(
            m_f32.nrows(),
            m_f32.ncols(),
            m_f32.iter().map(|v| N::from(*v) )
        );
        Ok(m)
    }

    pub fn into_file_or_stdout<N>(
         m : &DMatrix<N>,
         path : Option<&str>
    ) -> Result<(), ()>
        where N : Scalar + Display + FromStr
    {
        match path {
            Some(p) => {
                let strg_packed : String = build_string_packed(m);
                save_content_to_file(&strg_packed[..], p)
            },
            None => { print_packed(m); Ok(()) }
        }
    }

    /// Calls a function on a decoded matrix. If in_file is informed, read
    /// text from file, or block-wait for stdin if not path is informed.
    /// If Fn returns some result, write it to out_file, or to stdout otherwise.
    pub fn apply_filter<N> (
        func : &dyn Fn(&DMatrix<N>)->Option<DMatrix<N>>,
        in_file : Option<&str>,
        out_file : Option<&str>
    ) -> Result<(), ()>
        where N : Scalar + From<f32> + Display + FromStr
    {
        let m = from_file_or_stdin(in_file)?;
        let ans = func(&m).ok_or(())?;
        into_file_or_stdout(&ans, out_file)
    }

    pub fn apply_filters<N>(
        funcs : Vec<&dyn Fn(&DMatrix<N>)->Option<DMatrix<N>>>,
        in_file : Option<&str>,
        out_file : Option<&str>
    ) -> Result<(), ()>
        where N : Scalar + From<f32> + Display + FromStr
    {
        let m0 = Some(from_file_or_stdin(in_file)?);
        let final_ans = funcs.iter()
            .fold(m0, |opt_mtx, func| {
                match opt_mtx {
                    Some(m) => func(&m),
                    None => None
                }
            });
        match final_ans {
            Some(ref final_m) => into_file_or_stdout(final_m, out_file),
            None => Err(())
        }
    }

    /*pub fn filter_cli(
         func : &dyn Fn(&DMatrix<N>, HashMap<String, String>)->Option<DMatrix<N>>,
         required_params : HashMap<String, String>
    ) {
    }*/
}

use std::cell::RefCell;

pub struct CSVFilter<'a, N>
    where N : Scalar
{
    args : HashMap<String, Option<String>>,
    col_names : Vec<String>,
    fns : RefCell<HashMap<String, &'a dyn Fn(&DMatrix<N>)->Result<DMatrix<N>, String>>>
}

impl<'a, N> CSVFilter<'a, N>
    where N : Scalar
{

    pub fn new() -> Self {
        let mut args = HashMap::new();
        args.insert(String::from("Hello"), Some(String::from("Hello there")));
        CSVFilter {
            args,
            col_names : Vec::new(),
            fns : RefCell::new(HashMap::new())
        }
    }

    pub fn register(&self, name : &str, f : &'a dyn Fn(&DMatrix<N>)->Result<DMatrix<N>, String>) {
        self.fns.borrow_mut().insert(name.to_string(), f);
    }

    pub fn call(&self, name : &str, data : &DMatrix<N>) -> Result<DMatrix<N>, String> {
        (self.fns.borrow()[name])(data)
    }

    pub fn args(&self) -> HashMap<String, Option<String>> {
        self.args.clone()
    }

    /// Takes ownership of the CLI and run it once,
    /// reading all function calls that were made by the user and
    /// executing them.
    pub fn run(self) {

    }

}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    pub fn test_csv_filter() -> Result<(), String> {
        // Reading the command line arguments should already have happened
        // here, at instantiation, before users take reference to them
        // to pass them into his function
        let r = CSVFilter::new();
        let args = &r.args();
        let my_f = |a : &DMatrix<f64>|->Result<DMatrix<f64>, String> {
            println!("Hello {:?}", args);
            Ok(a.clone())
        };
        let my_f2 = |a : &DMatrix<f64>|->Result<DMatrix<f64>, String> {
            println!("Hello again {:?}", args);
            Ok(a.clone())
        };
        r.register("do this", &my_f);
        r.register("do that", &my_f2);
        r.call("do this", &DMatrix::<f64>::zeros(10,10))?;
        r.call("do that", &DMatrix::<f64>::zeros(10,10))?;
        Ok(())
    }

}*/

