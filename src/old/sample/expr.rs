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

#[derive(Debug, Clone)]
pub struct BinExpr {
    pub left : Value,
    pub op : String,
    pub right : Box<Expression>
}

pub enum BinPos {

    /// Invalid means that expression cannot be
    /// considered an associative chain.
    Invalid,

    /// Last means there is a final expression at the rhs, so the size
    /// of the associative chain is at least one.
    Last(Value),

    /// Middle means there is a nested binexpr after this one
    Middle(BinExpr)
}

impl BinExpr {

    fn bin_op(s : &str) -> IResult<&str, &str> {
        alt((
            tag("+"), tag("-"), tag("*"), tag("/"), tag("<"),
            tag(">"), tag(">="), tag("<="), tag(":"),
            tag("!="), tag("=="), tag("|")))(s)
    }

    pub fn parse(s : &str) -> IResult<&str, Self> {
        let ans = tuple((
            Value::parse,
            BinExpr::bin_op,
            Expression::parse
        ))(s)?;
        let left = (ans.1).0;
        let op = (ans.1).1.to_string();
        let right = Box::new( (ans.1).2 );
            Ok((ans.0, BinExpr{left, op, right}))
    }

    fn take_while_associate(bin_exprs : &mut Vec<BinExpr>) -> Option<Vec<BinExpr>> {
        match *bin_exprs.last()?.right.clone() {
            Expression::Bin(bin) => {
                bin_exprs.push(bin);
                BinExpr::take_while_associate(bin_exprs);
            },
            Expression::Val(_) => { },
            _ => { return None; }
        }
        Some(bin_exprs.to_vec())
    }

    /// Flatten a sequence of expressions that can be combined in an associative
    /// way. Returns the chain until the first non-associative expression is found.
    pub fn flatten_associative(&self) -> Option<Vec<BinExpr>> {
        let mut exprs = Vec::new();
        exprs.push(self.clone());
        BinExpr::take_while_associate(&mut exprs)
    }

    /// Returns a vector with all tokens in sequential order,
    /// if binary expression forms a valid associative chain.
    pub fn associative_chain(&self) -> Option<Vec<String>> {
        let chain = self.flatten_associative()?;
        let mut names = Vec::new();
        for expr in chain.iter().take(chain.len() - 1) {
            names.push(expr.left.to_string());
            names.push(expr.op.clone());
        }
        if let Some(e) = chain.last() {
            names.push(e.left.to_string());
            names.push(e.op.clone());
            if let Expression::Val(ref v) = *e.right {
                names.push(v.to_string());
            }
        }
        Some(names)
    }

    //pub fn reorder_precedence_children(&mut self) -> bool {
        /*let mut has_reordered = true;
        while has_reordered {
            while self.right == Expression::BinExpr(_) {

            }
        }*/
    //    true
    //}

    // Translate this binary expression to an FnCall expression,
    // which is ultimately how they will be found on the registry.
    /*pub fn as_fn_call(&self) -> FnCall {
        let name = self.op;
        let mut args = Vec::new();
        args.push(*self.left.clone());
        args.push(*self.right.clone());
        FnCall{ name, args }
    }*/
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

#[derive(Debug, Clone)]
pub enum Value {
    Float(f32),
    Integer(i32),
    Name(String)
}

impl Value {

    pub fn int_parser(s : &str) -> IResult<&str, i32> {
        match pair(digit1::<_, (_, ErrorKind)>, not(tag(".")))(s) {
            Ok((i, (d,_))) => Ok((i, d.parse::<i32>().unwrap())),
            Err(e) => Err(e)
        }
    }

    pub fn parse(s : &str) -> IResult<&str, Self> {
        match Value::int_parser(s) {
            Ok((i, d)) => Ok((i, Value::Integer(d))),
            Err(_) => match float::<_, (_, ErrorKind)>(s) {
                Ok((i, f)) => Ok((i, Value::Float(f))),
                Err(_) => match many1( alt((alphanumeric1, tag("_"))) )(s) {
                    Ok((i, v)) => {
                        let s : String = v.iter().map(|s| *s).collect();
                        Ok( (i, Value::Name(s.into())) )
                    },
                    Err(e) => Err(e)
                }
            }
        }
    }

}

impl Display for Value {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Float(v) => write!(f, "{}", v),
            Value::Integer(v) =>  write!(f, "{}", v),
            Value::Name(v) =>  write!(f, "{}", v),
        }
    }

}

#[derive(Debug, Clone)]
pub enum Expression {
    Val(Value),
    Bin(BinExpr),
    Fn(FnCall),
    Nested(Box<Expression>)
}

impl Expression {

    fn nested_expr(s : &str) -> IResult<&str, Expression> {
        delimited(tag("("), Expression::parse, tag(")"))(s)
    }

    pub fn parse(s : &str) -> IResult<&str, Self> {
        match BinExpr::parse(s) {
            Ok((i, b)) => Ok((i, Expression::Bin(b))),
            Err(_) => match FnCall::parse(s) {
                Ok((i, f)) => Ok((i, Expression::Fn(f))),
                Err(_) => match Value::parse(s) {
                    Ok((i, v)) => Ok((i, Expression::Val(v))),
                    Err(e) => Err(e)
                }
            }
        }
    }

    /// Returns the position of the current expression inside an associative
    /// chain by examining what the RHS is. Unless RHS is Value or another BinExpr,
    /// returns BinPos::Invalid.
    pub fn as_associative_pos(&self) -> BinPos {
        match self {
            Expression::Bin(bin) => {
                match *bin.right {
                    Expression::Val(ref v) => BinPos::Last(v.clone()),
                    Expression::Bin(ref bin) => BinPos::Middle(bin.clone()),
                    _ => BinPos::Invalid
                }
            },
            _ => BinPos::Invalid
        }
    }

    /*pub fn split_if_bin(&self) -> Option<(String, boxed::Box<Expression>)> {
        match self {
            Expression::Bin(bin) => Some( (bin.op.to_string(), bin.right) ),
            _ => None
        }
    }*/

    pub fn name_if_val(&self) -> Option<String> {
        match self {
            Expression::Val(v) => Some(v.to_string()),
            _ => None
        }
    }

    /*pub fn rhs_and_op_if_last_bin(&self) -> Option<(String, String)> {
        match self {
            Expression::Bin(bin) => {
                match *(bin.right) {
                    Expression::Val(v) => Some( (bin.op.to_string(), v.to_string()) ),
                    _ => None
                }
            }
        }
    }

    pub fn get_if_bin(&self) -> Option<BinExpr> {
        match self {
            Expression::Bin(bin) => Some(*bin),
            _ => None
        }
    }



    pub fn flat_to_str(&'a self, prev : Option<) -> Option<(String, Vec<(String, String)>)> {
        let mut compl : Vec<(String, String)> = Vec::new();
        let bin = self.split_if_bin()?;
        match self {

        }

        if let Some(name) = name_if_val(&self) {
            Some(name) => compl.push(name)
            _ => None
        }
    }*/

    //fn nest_binary_exprs(&self) -> Self {
        // Iterate over highest precedence operators (: and |).
        // Transform Bin(Bin(Bin) into Bin(Nested(Bin)).
    //}

}

#[derive(Debug, Clone)]
pub struct NamedExpression {
    pub name : Expression,
    pub expr : Expression
}

impl NamedExpression {

    pub fn new_from(s : &str) -> Result<Self, String> {
        let packed = NamedExpression::pack(s);
        let expr = separated_pair(
            Expression::parse,
            tag("="),
            Expression::parse
        )(&packed[..]);
        match expr {
            Ok((i, ex)) => {
                match i.len() {
                    0 => {
                        match ex.0 {
                            Expression::Val(_) | Expression::Fn(_) => {
                                Ok(Self{name : ex.0, expr : ex.1})
                            },
                            _ => {
                                Err("LHS is neither value nor function call".into())
                            }
                        }
                    },
                    _ => Err(format!("Invalid token: {}", i))
                }
            },
            Err(e) => Err(format!("Expression parsing error: {:?}", e))
        }
    }

    fn pack(expr : &str) -> String {
        expr.split(' ')
            .filter(|c| *c != "" && *c != "\t" && *c != "\n")
            .fold(String::new(), |acc, s| acc + s)
    }

    /// Returns a possible function call on first element
    /// and the name of the variable on the second element.
    pub fn lhs(&self) -> Result<(Option<String>, String), &'static str> {
        match &self.name {
            Expression::Fn(f) => {
                if let Some(a) = f.args.get(0) {
                    if let Expression::Val(v) = a {
                        Ok( (Some(f.name.clone()), v.to_string()) )
                    } else {
                        Err("LHS function argument is not value")
                    }
                } else {
                    Err("No args at LHS")
                }
            },
            Expression::Val(v) => Ok( (None, v.to_string()) ),
            _ => Err("Invalid LHS")
        }
    }

}

// Generate a slice from a data sequence over rows and cols
// slice(a,1:1,1:2)
// Cartesian product of discrete x discrete or discrete x continuous
// a:b
// eigv(mat(v, 5, 3)) Interpret column as matrix
// Decompose as basis functions
// spline(col, 3)
// wav(col, 3)

#[cfg(test)]
mod tests {

    use crate::expr::*;

    #[test]
    fn it_works() -> Result<(),()> {
        println!("{:?}", NamedExpression::new_from("a=b"));
        println!("{:?}", NamedExpression::new_from("a=b+c"));
        println!("{:?}", NamedExpression::new_from("a=b+c+d"));
        println!("{:?}", NamedExpression::new_from("a=f(b)"));
        println!("{:?}", NamedExpression::new_from("a=f(c,d)"));
        println!("{:?}", NamedExpression::new_from("a=1.0"));
        println!("{:?}", NamedExpression::new_from("a=f(1.0,2)"));
        println!("{:?}", NamedExpression::new_from("y=f(g(x))"));
        Ok(())
    }
}
