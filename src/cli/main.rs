use bayes::distr::*;
use structopt::*;
use bayes::sample::table::Table;
use std::default::Default;
use serde_json::{self, Value};
use std::convert::TryFrom;
use nalgebra::*;
use bayes::parse::AnyLikelihood;
use bayes::sample::Sample;
use bayes::inference;
use std::convert::TryInto;

/// Fit and compare probabilistic models from the command line
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

fn split_table(tbl : &Table) -> (DMatrix<f64>, DMatrix<f64>) {
    let y = tbl.at(1).unwrap().clone_owned();
    let x = multinormal::utils::append_intercept(
        tbl.at(0).unwrap().clone_owned()
    );
    (y, x)
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
                            let mle = model.mle(tbl.at((0, tbl.ncols())).unwrap())
                                .map_err(|e| format!("{}", e) )?;
                            print_or_save(mle, output)
                        },
                        "irls" => {
                            let mut bern : Bernoulli = model.try_into()?;
                            let (y, x) = split_table(&tbl);
                            let mn : &mut MultiNormal = bern.factor_mut().unwrap();
                            mn.scale_by(x.clone());
                            let eta = mn.mean().clone_owned();
                            bern.set_parameter((&eta).into(), false);
                            let bern_out = inference::irls(
                                bern,
                                y,
                                x
                            )?;
                            print_or_save(AnyLikelihood::Bern(bern_out), output)
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



