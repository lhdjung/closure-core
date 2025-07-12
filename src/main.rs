pub mod sprite;
pub mod sprite_types;
pub mod grimmer;

use rand::prelude::*;
use rand::rngs::StdRng;
use crate::sprite::{set_parameters, find_possible_distributions};
use crate::sprite_types::RestrictionsOption;

fn main() {
    let sprite_parameters = set_parameters(2.2, 1.3, 23, 1, 5, None, None, 1, None, RestrictionsOption::Default, false).unwrap();
    let results = find_possible_distributions(&sprite_parameters, 5, false, &mut StdRng::seed_from_u64(1234));

    assert_eq!(results[0].mean, 2.2);
}
