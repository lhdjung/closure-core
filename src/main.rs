use closure_core::sprite::{mean, sprite_parallel};
use closure_core::sprite_types::RestrictionsOption;
use rand::prelude::*;
use rand::rngs::StdRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(1234);

    let results: Vec<Vec<i32>> = sprite_parallel(
        2.2_f64,
        1.3_f64,
        23,
        1,
        5,
        None,
        None,
        1,
        5,
        None,
        RestrictionsOption::Default,
        false,
        &mut rng,
    )
    .unwrap();

    // Convert scaled integers to floats (scale factor is 10 for precision 1)
    let first_dist: Vec<f64> = results[0].iter().map(|&v| v as f64 / 10.0).collect();
    let computed_mean = mean(&first_dist);

    assert_eq!((computed_mean * 10.0).round() / 10.0, 2.2);
}
