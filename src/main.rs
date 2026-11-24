use closure_core::{sprite_parallel, RestrictionsOption};

fn main() {
    let result = sprite_parallel(
        2.2,  // mean
        1.3,  // sd
        23,   // n
        1,    // scale_min
        5,    // scale_max
        0.05, // rounding_error_mean
        0.05, // rounding_error_sd
        1,    // items
        None, // restrictions_exact
        RestrictionsOption::Default,
        None,    // parquet_config
        Some(5), // stop_after
        None,    // prominence
    )
    .unwrap();

    println!("Generated {} distributions", result.results.len());
    // stop_after is an upper bound: SPRITE may stop early with fewer
    assert!(!result.results.is_empty());
}
