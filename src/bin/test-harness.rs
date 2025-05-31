use std::env;
use std::fs::File;
use std::io;
use std::io::Result;

use csv::WriterBuilder;

use closure_core::dfs_parallel;

fn main() -> std::io::Result<()> {
    println!("Running test harness...");

    // Call library function to test
    write_closure_csv(3.5, 1.0, 120, 1, 5, 0.05, 0.05, "parallel_results.csv")?;

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

    // Compute results (only this part is timed)
    let result = dfs_parallel(
        mean,
        sd,
        n,
        scale_min.try_into().unwrap(),
        scale_max.try_into().unwrap(),
        rounding_error_mean,
        rounding_error_sd,
    );

    // Initialize CSV file
    let file = File::create(output_file)?;
    let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

    // Write header
    let header: Vec<String> = (1..=n).map(|i| format!("n{}", i)).collect();
    writer.write_record(&header)?;

    // Write combinations with progress updates
    let _chunk_size = if result.len() >= 100 {
        result.len() / 100 // Update every 1%
    } else {
        1 // Update for every combination if fewer than 100
    };

    for (_i, combination) in result.iter().enumerate() {
        writer.write_record(
            &combination
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>(),
        )?;
    }

    //bar.finish_with_message("Done!");

    println!("Number of valid combinations: {}", result.len());
    println!("Wrote result file: {}", cwd.clone() + "/" + output_file);

    Ok(())
}

fn get_current_dir() -> io::Result<String> {
    let path = env::current_dir()?;
    path.into_os_string()
        .into_string()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8 in path"))
}
