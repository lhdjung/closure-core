use closure_core;

fn main() -> std::io::Result<()> {
    println!("Running test harness...");
    
    // Call library function to test
    closure_core::write_closure_csv(
        5.0,
        2.78,
        30,
        1,
        10,
        0.01,
        0.01,
        "parallel_results.csv"
    )?;
    
    println!("Test completed successfully");
    Ok(())
}