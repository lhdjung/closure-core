// mod write_csv;
// use write_csv::write_closure_csv;

use std::env;
use std::fs::File;
use std::io;
use std::io::Result;

use csv::WriterBuilder;
use indicatif::{ProgressBar, ProgressStyle};

use closure_core::{count_initial_combinations, closure_parallel};

fn main() -> std::io::Result<()> {
    println!("Running test harness...");

    // Call library function to test
    write_closure_csv(3.5, 0.5, 52, 1, 5, 0.05, 0.05, "parallel_results.csv")?;

    println!("Test completed successfully");
    Ok(())
}

/// Write CLOSURE results to disk with progress tracking
fn write_closure_csv(
    mean: f64,
    sd: f64,
    n: usize,
    scale_min: i32,
    scale_max: i32,
    rounding_error_mean: f64,
    rounding_error_sd: f64,
    output_file: &str,
) -> Result<()> {
    let mut cwd = String::new();

    // Record working directory
    match get_current_dir() {
        Ok(path) => cwd = path,
        Err(e) => {
            eprintln!("Error getting current directory: {}", e);
        }
    }

    // Setup progress bar
    let n_usize = n as usize;
    let depth = (n_usize / 10).clamp(2, 15.min(n_usize - 1));
    let initial_count = count_initial_combinations(scale_min, scale_max, depth);
    let bar = ProgressBar::new(initial_count as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} Computing...")
            .unwrap()
            .progress_chars("=>-"),
    );

    // Compute results (only this part is timed)
    let result = closure_parallel(
        mean,
        sd,
        n,
        scale_min.try_into().unwrap(),
        scale_max.try_into().unwrap(),
        rounding_error_mean,
        rounding_error_sd,
        1,
        None,
        None,
    )
    .unwrap();

    // Initialize CSV file
    let file = File::create(output_file)?;
    let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

    // Write header
    let header: Vec<String> = (1..=n).map(|i| format!("n{}", i)).collect();
    writer.write_record(&header)?;

    // Write combinations with progress updates
    let chunk_size = if result.results.sample.len() >= 100 {
        result.results.sample.len() / 100 // Update every 1%
    } else {
        1 // Update for every combination if fewer than 100
    };

    for (i, combination) in result.results.sample.iter().enumerate() {
        writer.write_record(
            &combination
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>(),
        )?;

        if i % chunk_size == 0 {
            bar.inc(1);
        }
    }

    bar.finish_with_message("Done!");

    println!(
        "Number of valid combinations: {}",
        result.results.sample.len()
    );
    println!("Wrote result file: {}", cwd.clone() + "/" + output_file);

    Ok(())
}

fn get_current_dir() -> io::Result<String> {
    let path = env::current_dir()?;
    path.into_os_string()
        .into_string()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8 in path"))
}
