use closure_core::{sprite_parallel, RestrictionsOption};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(1234);

    let result = sprite_parallel(
        2.2,      // mean
        1.3,      // sd
        23,       // n
        1,        // scale_min
        5,        // scale_max
        0.05,     // rounding_error_mean
        0.05,     // rounding_error_sd
        1,        // n_items
        None,     // restrictions_exact
        RestrictionsOption::Default,
        false,    // dont_test
        None,     // parquet_config
        Some(5),  // stop_after
        &mut rng,
    )
    .unwrap();

    println!("Generated {} distributions", result.results.sample.len());
    // Note: stop_after is a suggestion, actual count may vary slightly
    assert!(result.results.sample.len() > 0);
}
