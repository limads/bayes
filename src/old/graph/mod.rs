use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::number::complete::float;
use nom::sequence::delimited;
use nom::sequence::tuple;
use nom::sequence::separated_pair;
use nom::sequence::pair;
use nom::multi::separated_list;
use nom::character::complete::{self, alpha1, alphanumeric1};
use std::boxed::Box;
use nom::character::complete::digit1;
use nom::IResult;
use nom::error::ErrorKind;
use nom::combinator::not;
use std::fmt::{self, Display};
use std::boxed;
use nom::multi::many1;

pub struct Graph {
    edges : Vec<Edge>
    nodes : Vec<Node>
}

pub enum NodeKind {
    Fixed,
    Random
}

pub struct Node {
    name : String
    op : Option<String>,
    kind : NodeKind
}

pub struct Edge {
    src : String,
    dst : String,
    corr : Option<f64>
}

impl Node {

    fn parse(s : &str) -> IResult<&str, Self> {
        let ans = pair(
            alpha1,
            delimited(
                tag("("),
                alpha1,
                tag(")")
            )(s)?

    }

}

#[derive(Debug, Clone)]
pub struct FnCall {
    name : String,
    args : Vec<Expression>
}

impl FnCall {

    pub fn parse(s : &str) -> IResult<&str, Self> {
        let ans = pair(
            alpha1,
            delimited(
                tag("("),
                separated_list(tag(","), Expression::parse),
                tag(")")
            )
        )(s)?;
        let name = (ans.1).0.to_string();
        let args = (ans.1).1;
        Ok((ans.0, FnCall{name, args}))
    }

}

separated_pair(
            Expression::parse,
            tag("="),
            Expression::parse
        )(&packed[..]);

match float::<_, (_, ErrorKind)>(s) {
                Ok((i, f)) => Ok((i, Value::Float(f))),
