use std::fs;
use std::error::Error;

struct Iris {
    sepal_length : Vec<f64>,
    sepal_width : Vec<f64>,
    petal_length : Vec<f64>,
    petal_width : Vec<f64>,
    species : Vec<String>
}

impl Iris {

    pub fn load() -> Result<Self, Box<dyn Error>> {
        let data = fs::read_to_string("examples/data/iris.csv")?;
        let mut sepal_length : Vec<f64> = Vec::new();
        let mut sepal_width : Vec<f64> = Vec::new();
        let mut petal_length : Vec<f64> = Vec::new();
        let mut petal_width : Vec<f64> = Vec::new();
        let mut species : Vec<String> = Vec::new();
        let mut parsed_fields = [
            &mut sepal_length,
            &mut sepal_width,
            &mut petal_length,
            &mut petal_width
        ];
        for row in data.split(|c| c == '\n' ).skip(1) {
            for (i, field) in row.split(|c| c == ',').enumerate() {s
                if i < 4 {
                    parsed_fields[i].push(field.parse::<f64>()?);
                } else {
                    species.push(field.to_string());
                }
            }
        }
        Ok(Self {
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            species
        })
    }

}

fn main() -> Result<(), Box<dyn Error>> {
    let iris = Iris::load()?;
    Ok(())
}


