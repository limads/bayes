use std::collections::HashMap;
use nalgebra::*;

#[derive(Debug, Clone)]
pub struct Column {
    pub name : String,
    content : ColumnContent
}

#[derive(Debug, Clone)]
pub enum ColumnContent {
    Numeric(Vec<f64>),
    Text(Vec<String>),
    Integer(Vec<i32>),
    Binary(Vec<Vec<u8>>)
}

impl ColumnContent {

    /// Recovers content as numeric. If content is not numeric, try to parse it from
    /// from Text or Integer variants. Always fail at binary variant.
    pub fn as_dbl_vec(&self) -> Result<Vec<f64>,&'static str> {
        match &self {
            ColumnContent::Text(col) => {
                let mut data = Vec::new();
                for s in col.iter() {
                    if let Ok(d) = (*s).parse::<f64>() {
                        data.push(d);
                    } else {
                        return Err("Could not parse entry as f64");
                    }
                }
                Ok(data)
            }
            ColumnContent::Numeric(n) => Ok(n.clone()),
            ColumnContent::Integer(n) => {
                Ok(n.iter().map(|i| *i as f64).collect::<Vec<_>>())
            },
            ColumnContent::Binary(_) => {
                Err("Could not parse entry as f64")
            }
        }
    }

    pub fn as_int_vec(&self) -> Result<Vec<i32>, &'static str> {
        match &self {
            ColumnContent::Text(col) => {
                let mut data = Vec::new();
                for s in col.iter() {
                    if let Ok(d) = (*s).parse::<i32>() {
                        data.push(d);
                    } else {
                        return Err("Could not parse entry as i32");
                    }
                }
                Ok(data)
            },
            ColumnContent::Integer(col) => {
                Ok(col.clone())
            }
            _ => Err("Could not parse entry as i32")
        }
    }

    pub fn from_num(v : Vec<f64>) -> Self {
        ColumnContent::Numeric(v)
    }


}

impl<'a> Column {

    /// Tries to parse v in the order i32 -> f64 -> Text. Always succeeds at text.
    /// Result is packed into a variant of the first type that could be successfully parsed.
    pub fn from_vec_try_integer(name : String, v : Vec<String>) -> Self {
        let col = ColumnContent::Text(v.clone());
        match col.as_int_vec() {
            Ok(n) => Self{ name, content : ColumnContent::Integer(n) },
            Err(_) => Self::from_vec_try_num(name, v)
        }
    }

    /// Try to parse v in the order f64 -> Text. Always succeeds as text.
    /// Result is packed into a variant of the first type that could be successfully parsed.
    pub fn from_vec_try_num(name : String, v : Vec<String>) -> Self {
        let col = ColumnContent::Text(v);
        match col.as_dbl_vec() {
            Ok(n) => {
                /*println!("column parsed as f64");*/
                Self { name, content : ColumnContent::Numeric(n) }
            },
            Err(e) => {
                println!("{}", e);
                Self { name : name, content : col }
            }
        }
    }

    /// Builds as a numeric variant.
    pub fn new_num(name : String, data : Vec<f64>) -> Self {
        Column {
            name,
            content : ColumnContent::Numeric(data)
        }
    }

    /// Builds as a text variant.
    pub fn new_text(name : String, data : Vec<String>) -> Self {
        Column {
            name,
            content : ColumnContent::Text(data)
        }
    }

    /// Get content as f64 if it is f64 or i32. Return None otherwise.
    pub fn get_if_numeric(&self) -> Option<Vec<f64>> {
        match &self.content {
            ColumnContent::Numeric(n) => Some(n.clone()),
            ColumnContent::Integer(n) => {
                Some(n.iter().map(|i| *i as f64).collect::<Vec<_>>())
            }
            _ => None
        }
    }

    /// Get content as i32 iff content is i32.
    pub fn get_if_integer(&self) -> Option<Vec<i32>> {
        match &self.content {
            ColumnContent::Integer(n) => Some(n.clone()),
            _ => None
        }
    }

    /// Get content as text. Always successful, although String will carry
    /// no useful information if column is binary. It will just contain
    /// the '(Binary)' placeholder in this case, since there is no inherent
    /// guarantee that the binary content is valid UTF-8.
    pub fn as_string_vec(&self) -> Vec<String> {
        match &self.content {
            ColumnContent::Text(c) => c.clone(),
            ColumnContent::Numeric(n) => n.iter().map(|n| n.to_string()).collect(),
            ColumnContent::Integer(n) => n.iter().map(|n| n.to_string()).collect(),
            ColumnContent::Binary(b) => b.iter().map(|_| String::from("(Binary)")).collect()
        }
    }

    /// Returns index as string, if it exists. Will carry no information if column
    /// is binary (will return '(Binary)' placeholder if index is valid but content
    /// is binary.
    pub fn index_as_string(&self, ix : usize) -> Option<String> {
        match &self.content {
            ColumnContent::Text(c) => c.get(ix).map(|e| e.clone()),
            ColumnContent::Numeric(n) => n.get(ix).map(|e| e.to_string()),
            ColumnContent::Integer(n) => n.get(ix).map(|e| e.to_string()),
            ColumnContent::Binary(b) => b.get(ix).map(|_| String::from("(Binary)"))
        }
    }

    /// Textual description of the column content that matches SQLite3's types.
    /// Useful for building table creation SQL statements. Will also work
    /// with PostgreSQL. For reference, see https://www.sqlite.org/datatype3.html
    pub fn sqlite3_type(&self) -> String {
        match self.content {
            ColumnContent::Text(_) => String::from("text"),
            ColumnContent::Integer(_) => String::from("integer"),
            ColumnContent::Numeric(_) => String::from("real"),
            ColumnContent::Binary(_) => String::from("blob")
        }
    }

    /// Returns the length of the column irrespective of the content.
    pub fn len(&self) -> usize {
        match &self.content {
            ColumnContent::Text(c) => c.len(),
            ColumnContent::Integer(c) => c.len(),
            ColumnContent::Numeric(c) => c.len(),
            ColumnContent::Binary(c) => c.len()
        }
    }

    /// Iterate over a column it it contains textual content; Returns None otherwise.
    pub fn text_iter(&'a self) -> Option<impl Iterator<Item=&'a str>> {
        match &self.content {
            ColumnContent::Text(tc) => {
                Some(tc.iter().map(|s| &s[..] ))
            },
            _ => None
        }
    }
}

