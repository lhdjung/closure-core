pub mod grimmer;
pub mod sprite;
pub mod sprite_types;

use crate::sprite::{find_possible_distributions, set_parameters};
use crate::sprite_types::RestrictionsOption;
use rand::prelude::*;
use rand::rngs::StdRng;

fn main() {
    let sprite_parameters = set_parameters(
        2.2,
        1.3,
        23,
        1,
        5,
        None,
        None,
        1,
        None,
        RestrictionsOption::Default,
        false,
    )
    .unwrap();
    let results = find_possible_distributions(
        &sprite_parameters,
        5,
        false,
        &mut StdRng::seed_from_u64(1234),
    );

    assert_eq!(results[0].mean, 2.2);
}
