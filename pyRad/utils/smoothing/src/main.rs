//Dose smoothing module in Rust!

//Copyright Marc-Andre Renaud, 2017
extern crate smoothing;

use std::path::Path;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let filename = args[1].clone();
    let dose_path = Path::new(&filename);
    let dose = smoothing::dose::mc_dose::MCDose::from_file(dose_path);

    println!("Done loading dose, performing smoothing");

    let smoothed_dose = dose.smooth();
    let splitted_path : Vec<&str> = filename.split('.').collect();
    let new_filename = splitted_path[0].to_string() + "_smoothed.3ddose";
    smoothed_dose.write_3ddose(new_filename);
}