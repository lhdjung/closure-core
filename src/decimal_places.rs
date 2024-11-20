use std::num::ParseFloatError;

fn count_decimal_places(num_str: &str) -> Result<usize, ParseFloatError> {
    // First verify the string can be parsed as a number
    num_str.parse::<f64>()?;
    
    // Split the string by decimal point
    let parts: Vec<&str> = num_str.split('.').collect();
    
    // If there's no decimal point or it's the last character
    if parts.len() == 1 || parts[1].is_empty() {
        return Ok(0);
    }
    
    // Count the digits after decimal point
    let decimal_part = parts[1];
    Ok(decimal_part.len())
}


pub fn rounding_error(num_str: &str) -> f64 {
    let decimal_places = count_decimal_places(num_str);
    1.0 / 10.0_f64.powi(decimal_places.unwrap() as i32)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_numbers() {
        assert_eq!(count_decimal_places("123.456").unwrap(), 3);
        assert_eq!(count_decimal_places("0.1").unwrap(), 1);
        assert_eq!(count_decimal_places("123").unwrap(), 0);
        assert_eq!(count_decimal_places("123.0").unwrap(), 1);
        assert_eq!(count_decimal_places("123.4500").unwrap(), 4);
        assert_eq!(count_decimal_places("-123.456").unwrap(), 3);
    }

    #[test]
    fn test_invalid_numbers() {
        assert!(count_decimal_places("abc").is_err());
        assert!(count_decimal_places("12.34.56").is_err());
        assert!(count_decimal_places("").is_err());
    }
}