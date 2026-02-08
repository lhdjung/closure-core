/// Determine whether the two provided numbers are within a given tolerance of each other
pub fn is_near(num_1: f64, num_2: f64, tolerance: f64) -> bool {
    (num_1 - num_2).abs() <= tolerance
}

/// rust does not have a native function that rounds binary floating point numbers to a set number
/// of decimals. This is a hacky workaround that nevertheless seems to be the best option
pub fn round_f64(x: f64, y: i32) -> f64 {
    (x * 10.0f64.powi(y)).round() / 10.0f64.powi(y)
}
