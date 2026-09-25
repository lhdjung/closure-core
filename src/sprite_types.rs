use std::collections::hash_map::Keys;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct OccurrenceConstraints {
    /// Must have exactly this many of each value
    pub exact: HashMap<i32, usize>,
    /// Must have at least this many of each value
    pub minimum: HashMap<i32, usize>,
}

impl OccurrenceConstraints {
    pub fn new(exact: HashMap<i32, usize>, minimum: HashMap<i32, usize>) -> Self {
        Self { exact, minimum }
    }

    pub fn check_conflicts(&self) -> bool {
        let exact_keys: std::collections::HashSet<_> = self.exact.keys().collect();
        let min_keys: std::collections::HashSet<_> = self.minimum.keys().collect();

        !exact_keys.is_disjoint(&min_keys)
    }
}

/// Minimum counts for specific scale values: at least `count` responses at
/// each key.
///
/// Keys are in **hundredths of a scale point**, like every restriction key in
/// SPRITE: `300` is the value 3 and `133` is 1 1/3 on a three-item scale. A key
/// that is not a grid value is an input error.
#[derive(Debug, Clone)]
pub struct RestrictionsMinimum(pub HashMap<i32, usize>);

impl RestrictionsMinimum {
    /// At least one response at each of `min` and `max`, given as scale
    /// values: `from_range(1, 5)` for a 1-5 scale. The keys it stores are in
    /// hundredths like all others.
    pub fn from_range(min: i32, max: i32) -> Self {
        let mut map = HashMap::new();
        // Saturating, so an absurd scale gives an invalid key, not a panic.
        map.insert(min.saturating_mul(100), 1);
        map.insert(max.saturating_mul(100), 1);
        Self(map)
    }

    pub fn keys(&self) -> Keys<'_, i32, usize> {
        self.0.keys()
    }

    pub fn keys_rounded(&self) -> impl Iterator<Item = f64> + '_ {
        self.0.keys().map(|&k| (k as f64).round())
    }

    pub fn new(hashmap: HashMap<i32, usize>) -> Self {
        Self(hashmap)
    }

    pub fn extract(&self) -> HashMap<i32, usize> {
        self.clone().0
    }
}

/// Which minimum-count restrictions SPRITE applies.
#[derive(Debug, Clone)]
pub enum RestrictionsOption {
    /// At least one response at `scale_min` and one at `scale_max`, i.e. the
    /// reported scale is also the observed range. This excludes every sample
    /// that does not reach both ends, and makes an exact restriction on
    /// either end a conflict; use [`RestrictionsOption::Null`] for none.
    Default,
    /// The given minimum counts, or none for `Opt(None)`.
    Opt(Option<RestrictionsMinimum>),
    /// No minimum counts.
    Null,
}

impl RestrictionsOption {
    pub fn is_default(&self) -> bool {
        match self {
            RestrictionsOption::Default => true,
            RestrictionsOption::Opt(_) => false,
            RestrictionsOption::Null => false,
        }
    }

    pub fn is_null(&self) -> bool {
        match self {
            RestrictionsOption::Opt(s) => s.is_none(),
            RestrictionsOption::Default => false,
            RestrictionsOption::Null => true,
        }
    }

    pub fn new(restrictions_minimum: RestrictionsMinimum) -> Self {
        RestrictionsOption::Opt(Some(restrictions_minimum))
    }

    pub fn new_default() -> Self {
        RestrictionsOption::Default
    }

    /// [`RestrictionsMinimum::from_range`] for scale values `min` and `max`.
    pub fn construct_from_default(self, min: i32, max: i32) -> Self {
        RestrictionsOption::Opt(Some(RestrictionsMinimum::from_range(min, max)))
    }

    pub fn extract(self) -> RestrictionsMinimum {
        match self {
            RestrictionsOption::Opt(s) => s.unwrap(),
            _ => panic!(),
        }
    }
}
