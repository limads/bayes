use bayes::distr::*;
use structopt::*;
use bayes::sample::table::Table;
use std::default::Default;
use serde_json::{self, Value};
use std::convert::TryFrom;
use nalgebra::*;
use bayes::parse::AnyLikelihood;
use bayes::sample::Sample;

#[derive(StructOpt, Debug)]
pub enum Bayes {

    /// Builds a variable graph from a probabilistic model (which can be a prior or posterior)
    Graph {
        src : Option<String>,

        #[structopt(short)]
        output : Option<String>,
    },

    /// Builds a probabilistic model from a variable graph (which can be a prior or posterior)
    Model {
        src : Option<String>,

        #[structopt(short)]
        output : Option<String>,
    },

    /// Fit a probabilistic model, with a posterior distribution as output, and potential
    /// trajectory file to monitor sampling/optimization trajectory.
    Fit {
        model : Option<String>,

        #[structopt(short)]
        data : Option<String>,

        #[structopt(short)]
        method : String,

        #[structopt(short)]
        output : Option<String>,

        #[structopt(short)]
        traj : Option<String>
    },

    /// Displays a table summary from the model output.
    Summary {
        src : Option<String>,

        #[structopt(short)]
        output : Option<String>
    }

}

fn open_table(src : &str) -> Result<Table, String> {
    match Table::open(src, Default::default()) {
        Ok(tbl) => Ok(tbl),
        Err(e) => Err(format!("Error opening table: {}", e))
    }
}

fn print_or_save(lik : AnyLikelihood, opt_path : &Option<String>) -> Result<(), String> {
    match opt_path {
        Some(path) => lik.save_to_path(path).map_err(|e| format!("{}", e)),
        None => { println!("{}", lik); Ok(()) }
    }
}

fn main() -> Result<(), String> {
    let bayes = Bayes::from_args();
    match &bayes {
        Bayes::Fit{ model, data, method, output, traj  } => {
            match (model, data) {
                (Some(model_path), Some(data_path)) => {
                    let tbl = open_table(data_path)?;
                    let model = AnyLikelihood::load_from_path(model_path)
                        .map_err(|e| format!("{}", e) )?;
                    match &method[..] {
                        "mle" => {
                            let mle = model.mle(tbl.at((0, tbl.ncols())).unwrap());
                            print_or_save(mle, output)
                        },
                        m => Err(format!("Unknown method: {}", m)),
                    }
                },
                (_, None) => {
                    Err("Missing data file".into())
                },
                (None, _) => {
                    Err("Missing model file".into())
                }
            }
        },
        _ => Err("Unimplemented option".into())
    }
}



