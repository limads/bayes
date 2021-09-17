// use bayes::distr::*;
use structopt::*;
// use bayes::sample::table::Table;
use std::default::Default;
use serde_json::{self, Value};
use std::convert::TryFrom;
use nalgebra::*;
// use bayes::model::Model;
use bayes::sample::Sample;
// use bayes::inference;
use std::convert::TryInto;
// use indicatif;
use std::thread;
use std::time::Duration;
use polars::prelude::CsvReader;
use bayes::prob::*;
use polars::prelude::SerReader;
use std::fs::File;
use std::io::Read;
use bayes::fit::Estimator;
use std::io;
use polars::frame::DataFrame;

// use warp::Filter;

/// Fit and compare probabilistic models from the command line
#[derive(StructOpt, Debug)]
pub enum Bayes {

    /* Serve {
    
        #[structopt(short)]
        host
    },*/
    
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
        #[structopt(short)]
        model : Option<String>,

        #[structopt(short)]
        data : Option<String>,

        // Outpt result as CSV
        #[structopt(short)]
        tabulate : bool

        /*#[structopt(short)] rename to algorithm?
        method : String,

        #[structopt(short)]
        output : Option<String>,

        #[structopt(short)]
        traj : Option<String>*/
    },

    /// Displays a table summary from the model output.
    Summary {
        src : Option<String>,

        #[structopt(short)]
        output : Option<String>
    }

}

/*fn open_table(src : &str) -> Result<Table, String> {
    match Table::open(src, Default::default()) {
        Ok(tbl) => Ok(tbl),
        Err(e) => Err(format!("Error opening table: {}", e))
    }
}*/

/*fn print_or_save(lik : Model, opt_path : &Option<String>) -> Result<(), String> {
    match opt_path {
        Some(path) => lik.save_to_path(path).map_err(|e| format!("{}", e)),
        None => { println!("{}", lik); Ok(()) }
    }
}*/

/*fn split_table(tbl : &Table) -> (DMatrix<f64>, DMatrix<f64>) {
    let y = tbl.at(1).unwrap().clone_owned();
    let x = multinormal::utils::append_intercept(
        tbl.at(0).unwrap().clone_owned()
    );
    (y, x)
}*/

/*async fn serve() -> Result<(), String> {
    /*GET /model/what 
    Host: hyper.rs
    User-Agent: reqwest/v0.8.6*/
    let model_route = warp::path("model")
        .and(warp::path::param())
        .and(warp::header("user-agent"))
        .map(|param: String, model: String| {
            format!("Hello {}, whose agent is {}", param, model)
        });
    Ok(())
}*/
//fn serve(host : &str) -> Result<(), String> {
//    Ok(())
//}

pub enum UnivariateLikelihood {
    Normal(Normal),
    Bernoulli(Bernoulli),
    Poisson(Poisson)
}

pub enum UnivariatePosterior {
    Normal(Normal),
    Gamma(Gamma),
    Beta(Beta)
}

/*fn fit_mle<L, O>(model : Value, data : &DataFrame) -> Result<(), Box<dyn Error>>
    where L : Distribution + Likelihood<O> + TryFrom<Value>
{
    lik : L = model.try_into()?;
    assert!(lik.view_factors().next().is_none());
    Ok(())
}

fn fit_conjugate<L, O, P>()
    where
        L : Likelihood<O> + TryFrom<Value>,
        P : Prior
{

}*/

fn main() -> Result<(), String> {
    let bayes = Bayes::from_args();
    match &bayes {
        /*Bayes::Serve { host } {
            serve(&host)
        },*/
        Bayes::Fit{ model, data, /*method, output, traj*/ tabulate  } => {

            // indicatif::ProgressIterator::progress((0..100).for_each(|v| { thread::sleep(Duration::from_millis(500)); v } )).msg("el");

            match (model, data) {
                (Some(model_path), Some(data_path)) => {

                    let df = CsvReader::from_path(&data_path).map_err(|e| format!("{}", e) )?
                        .infer_schema(None)
                        .has_header(true)
                        .finish().map_err(|e| format!("{}", e) )?;

                    println!("{}", df);

                    let mut f = File::open(model_path).map_err(|e| format!("{}", e) )?;
                    let mut s = String::new();
                    f.read_to_string(&mut s);

                    let model : Value = s.parse().map_err(|e| format!("{}", e) )?;
                    println!("{}", model);

                    for name in df.get_column_names().iter() {
                        if let Some(distr) = model.get(name) {
                            if let Ok(series) = df.column(name) {
                                let m : Normal = distr.clone().try_into().map_err(|e| format!("Invalid distribution"))?;

                                let data : Vec<f64> = series.f64().map_err(|e| format!("Series is not f64") )?
                                    .into_iter()
                                    .filter_map(|n| n )
                                    .collect();

                                // If user informed just "variable" : {} or ["var1, "var2"] (empty parameters),
                                // perform maximum likelihood inference. Perform conjugate estimation otherwise.
                                // If user informed "var1" : { "joint : ["var2", "var3"] } or "var1" : { "joint" : { "var2" : { "joint" : "var3" } } }
                                // Perform MLE estimation over the bayesian network.
                                let mut lik = Normal::likelihood(data.iter())
                                    .condition(m);

                                let post : Result<&Normal, _> = lik.fit(None);
                                match post {
                                    Ok(post) => {
                                        let decl = NormalDecl::from(post.clone());
                                        if *tabulate {
                                            let mut wtr = csv::Writer::from_writer(io::stdout());
                                            wtr.write_record(&["var", "location", "scale"]).map_err(|e| format!("{}", e) )?;
                                            wtr.write_record(&["name", &decl.mean.to_string(), &decl.var.to_string()])
                                                .map_err(|e| format!("{}", e) )?;
                                        } else {
                                            println!("{:?}", decl);
                                        }
                                    },
                                    Err(e) => {
                                        eprintln!("{:?}", e);
                                    }
                                }
                            } else {
                                return Err(format!("No column named {} at dataframe", name));
                            }
                        } else {
                            // Ignore variables not on model
                        }
                    }

                    Ok(())

                    /*let tbl = open_table(data_path)?;
                    let model = Model::load_from_path(model_path)
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
                            print_or_save(Model::Bern(bern_out), output)
                        },
                        m => Err(format!("Unknown method: {}", m)),
                    }*/
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



