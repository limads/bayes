use postgres::{self, row::Row, Client, /*types::FromSql, types::ToSql,*/ types::Type };
use sqlparser::dialect::PostgreSqlDialect;
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
        ix : usize,
        ignored_rows : &mut Vec<usize>
   ) -> Result<(), String> {
        for (i, e) in r_slice.iter_mut().enumerate() {
            match col_types[i] {
                ColumnType::Integer => {
                     match row.try_get::<usize, Option<i32>>(i).map_err(|e| format!("{}", e))? {
                        Some(val) => { *e = val as f64 },
                        None => { ignored_rows.push(ix); }
                     }
                },
                ColumnType::Long => {
                    match row.try_get::<usize, Option<i64>>(i).map_err(|e| format!("{}", e))? {
                        Some(val) => { *e = val as f64 },
                        None => { ignored_rows.push(ix); }
                     }
                },
                ColumnType::Double => {
                    match row.try_get::<usize, Option<f64>>(i).map_err(|e| format!("{}", e))? {
                        Some(val) => { *e = val },
                        None => { ignored_rows.push(ix); }
                    }
                },
                ColumnType::Float => {
                    match row.try_get::<usize, Option<f32>>(i).map_err(|e| format!("{}", e))? {
                        Some(val) => { *e = val as f64 },
                        None => { ignored_rows.push(ix);}
                    }
                },
                ColumnType::Boolean => {
                    let b = match row.try_get::<usize, Option<bool>>(i).map_err(|e| format!("{}", e))? {
                        Some(val) => { val },
                        None => { ignored_rows.push(ix); false }
                    };
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

    fn load_postgre(mut client : Client, sql : &str, max : usize, null : NullAction) -> Result<Self, String> {
        let dialect = PostgreSqlDialect {};
        let ast = Parser::parse_sql(&dialect, sql.to_string()).map_err(|e| format!("{}", e))?;
        if ast.len() != 1 {
            return Err(format!("Multiple statements passed"));
        }
        match ast[0] {
            Statement::Query(_) => { },
            _ => { return Err(format!("Non-select SQL statement passed")); }
        };
        let mut ignored_rows = Vec::new();
        match client.query(sql, &[]) {
            Ok(rows) => {
                if let Some(row1) = rows.get(0) {
                    let ncols = row1.len();
                    let mut data = DMatrix::zeros(max, ncols);
                    let col_names : Vec<String> = row1.columns().iter().map(|c| c.name().to_string()).collect();
                    let col_types = Self::column_types(row1).ok_or(format!("Invalid column type"))?;
                    Self::write_row(row1, data.row_mut(0), &col_types[..], 0, &mut ignored_rows)?;
                    let mut ix = 1;
                    while let Some(row) = rows.iter().skip(1).next() {
                        Self::write_row(row, data.row_mut(ix), &col_types[..], ix, &mut ignored_rows)?;
                        ix += 1;
                        if ix > max {
                            return Err(format!("Too many records returned"));
                        }
                    }
                    let valid_data = match (null, ignored_rows.len()) {
                        (NullAction::IgnoreRow, 0) => {
                            data
                        },
                        (NullAction::IgnoreRow, n_ignored) => {
                            let mut data_ix = 0;
                            let mut ignored_ix = 0;
                            let mut valid_data = DMatrix::zeros(ix - n_ignored, ncols);
                            for mut row in valid_data.row_iter_mut() {
                                if data_ix != ignored_rows[ignored_ix] {
                                    row.copy_from(&data.row(data_ix));
                                } else {
                                    ignored_ix += 1;
                                }
                                data_ix += 1;
                            }
                            valid_data
                        },
                        (NullAction::Error, 0) => {
                            data
                        },
                        (NullAction::Error, n_ignored) => {
                            return Err(format!("{} null values found", n_ignored));
                        }
                    };
                    let full_nrows = valid_data.nrows();
                    let trim_data = valid_data.remove_rows(ix, full_nrows);
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

    pub fn load<S>(source : S, query : &str, max : usize, null : NullAction) -> Result<Self, String>
        where S : Into<TableSource>
    {
        let source = source.into();
        match source {
            TableSource::Postgre(client) => Self::load_postgre(client, query, max, null),
            TableSource::File(f) => Self::load_from_file(f, null),
            _ => Err(format!("Unknown client"))
        }

    }
}








