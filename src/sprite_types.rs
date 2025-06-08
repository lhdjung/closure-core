use std::collections::HashMap;
use std::collections::hash_map::Keys;

/// references, clone copy
///
///
#[derive(Debug, Clone)]
pub struct OccurrenceConstraints {
    /// Must have exactly this many of each value
    pub exact: HashMap<i32, usize>,
    /// Must have at least this many of each value
    pub minimum: HashMap<i32, usize>,
    /// Must have at most this many of each value (optional)
    pub maximum: Option<HashMap<i32, usize>>,
}

impl OccurrenceConstraints {
    pub fn new(exact: HashMap<i32, usize>, minimum: HashMap<i32, usize>, maximum: Option<HashMap<i32, usize>>) -> Self {
        Self {
            exact, 
            minimum,
            maximum,
        }
    }

    pub fn check_conflicts(&self) -> bool {
        let exact_keys: std::collections::HashSet<_> = self.exact.keys().collect();
        let min_keys: std::collections::HashSet<_> = self.minimum.keys().collect();

        !exact_keys.is_disjoint(&min_keys)
    }
}

#[derive(Debug, Clone)]
pub struct RestrictionsMinimum(pub HashMap<i32, usize>);

impl RestrictionsMinimum {
    pub fn from_range(min: i32, max: i32) -> Self {
        let mut map = HashMap::new();
        map.insert(min, 1);
        map.insert(max, 1);
        Self(map)
    }

    pub fn keys(&self) -> Keys<'_, i32, usize> {
        self.0.keys()
    }
    // pub fn keys(&self) -> impl Iterator<Item = i32> + '_ {
    //     self.0.keys().map(|&k| k)
    // }

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

#[derive(Debug, Clone)]
pub enum RestrictionsOption {
    Default(),
    Opt(Option<RestrictionsMinimum>),
    Null(),
}

impl RestrictionsOption {
    pub fn is_default(&self) -> bool {
        match self {
            RestrictionsOption::Default() => true,
            RestrictionsOption::Opt(_) => false,
            RestrictionsOption::Null() => false,
        }
    }

    pub fn is_null(&self) -> bool {
        match self {
            RestrictionsOption::Opt(s) => s.is_none(),
            RestrictionsOption::Default() => false,
            RestrictionsOption::Null() => true,
        }
    }

    pub fn new(restrictions_minimum: RestrictionsMinimum) -> Self {
        RestrictionsOption::Opt(Some(restrictions_minimum))
    }

    pub fn new_default() -> Self {
        RestrictionsOption::Default()
    }

    pub fn construct_from_default(self, min: i32, max: i32) -> Self {
        RestrictionsOption::Opt(Some(RestrictionsMinimum::from_range(min, max)))
    }

    pub fn extract(self) -> RestrictionsMinimum {
        match self {
            RestrictionsOption::Opt(s) => s.unwrap(),
            _ => panic!()
        }
    }
}


