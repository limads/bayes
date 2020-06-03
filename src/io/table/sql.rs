use postgres::{self, row::Row, Client, /*types::FromSql, types::ToSql,*/ types::Type };
// use std::ops::Index;
// use std::ops::Range;
// use std::fs::{File};
///use std::path::Path;
// use std::convert::AsRef;
// use std::io;
use sqlparser::dialect::PostgreSqlDialect;
// use sqlparser::parser::Parser;
// use sqlparser::ast::Statement;
// use rust_decimal::Decimal;
use nalgebra::*;

use super::*;

impl Table {

    fn column_types(row : &Row) -> Option<Vec<ColumnType>> {
        let mut col_types = Vec::new();
        for col in row.columns().iter() {
            if *col.type_() == Type::BOOL {
                col_types.push(ColumnType::Boolean);
            } else {
                if *col.type_() == Type::FLOAT8 {
                    col_types.push(ColumnType::Double);
                } else {
                    if *col.type_() == Type::FLOAT4 {
                        col_types.push(ColumnType::Float);
                    } else {
                        if *col.type_() ==  Type::INT4 {
                            col_types.push(ColumnType::Integer);
                        } else {
                            if *col.type_() == Type::INT8 {
                                col_types.push(ColumnType::Long);
                            } else {
                                return None;
                                /*if col.type_() == Type::NUMERIC {
                                    col_types.push(ColumnType::Numeric);
                                } else {
                                    return None;
                                }*/
                            }
                        }
                    }
                }
            }
        }
        Some(col_types)
    }

    fn write_row(
        row : &Row,
        mut r_slice : MatrixSliceMut<f64, U1, Dynamic, U1, Dynamic>,
        col_types : &[ColumnType],
   ) -> Result<(), String> {
        for (i, e) in r_slice.iter_mut().enumerate() {
            match col_types[i] {
                ColumnType::Integer => {
                    *e = row.try_get::<usize, i32>(i).map_err(|e| format!("{}", e))? as f64;
                },
                ColumnType::Long => {
                    *e = row.try_get::<usize, i64>(i).map_err(|e| format!("{}", e))? as f64;
                },
                ColumnType::Double => {
                    *e = row.try_get::<usize, f64>(i).map_err(|e| format!("{}", e))?;
                },
                ColumnType::Float => {
                    *e = row.try_get::<usize, f32>(i).map_err(|e| format!("{}", e))? as f64;
                },
                ColumnType::Boolean => {
                    let b = row.try_get::<usize, bool>(i).map_err(|e| format!("{}", e))?;
                    match b {
                        true => *e = 1.0,
                        false => *e = 0.0
                    }
                },
                ColumnType::Numeric => {
                    return Err(format!("Numeric conversion unimplemented"));
                    /*let dec = r.try_get::<Decimal, usize>(i).map_err(|e| format!("{}", e))?;
                    if let Some(d) = dec.to_f64() {
                        *e = d;
                    } else {
                        return Err(format!("Invalid decimal conversion"));
                    },*/
                }
            }
        }
        Ok(())
    }

    fn load_postgre(mut client : Client, sql : &str, max : usize) -> Result<Self, String> {
        let dialect = PostgreSqlDialect {};
        let ast = Parser::parse_sql(&dialect, sql.to_string()).map_err(|e| format!("{}", e))?;
        if ast.len() != 1 {
            return Err(format!("Multiple statements passed"));
        }
        match ast[0] {
            Statement::Query(_) => { },
            _ => { return Err(format!("Non-select SQL statement passed")); }
        };
        match client.query(sql, &[]) {
            Ok(rows) => {
                if let Some(row1) = rows.get(0) {
                    let ncols = row1.len();
                    let mut data = DMatrix::zeros(max, ncols);
                    let col_names : Vec<String> = row1.columns().iter().map(|c| c.name().to_string()).collect();
                    let col_types = Self::column_types(row1).ok_or(format!("Invalid column type"))?;
                    Self::write_row(row1, data.row_mut(0), &col_types[..])?;
                    let mut ix = 1;
                    while let Some(row) = rows.iter().skip(1).next() {
                        Self::write_row(row, data.row_mut(ix), &col_types[..])?;
                        ix += 1;
                        if ix > max {
                            return Err(format!("Too many records returned"));
                        }
                    }
                    let full_nrows = data.nrows();
                    let trim_data = data.remove_rows(ix, full_nrows);
                    Ok(Self {
                        _source : TableSource::Postgre(client),
                        data : trim_data,
                        col_names,
                        col_types,
                    })
                } else {
                    Err(format!("No rows available"))
                }
            },
            Err(e) => Err(format!("{}", e))
        }
    }

    pub fn load<S>(source : S, query : &str, max : usize) -> Result<Self, String>
        where S : Into<TableSource>
    {
        let source = source.into();
        match source {
            TableSource::Postgre(client) => Self::load_postgre(client, query, max),
            TableSource::File(f) => Self::load_from_file(f),
            _ => Err(format!("Unknown client"))
        }

    }
}








