//! Shared benchmark infrastructure: the `Case` struct and the canonical case list.

pub struct Case {
    pub mean: f64,
    pub sd: f64,
    pub n: i32,
    pub scale_min: i32,
    pub scale_max: i32,
    pub re_mean: f64,
    pub re_sd: f64,
}

impl Case {
    /// `"n=20,  [1,7]"`, with a trailing `*` for wider-than-default rounding tolerance.
    pub fn label(&self) -> String {
        let star = if self.re_mean > 0.05 || self.re_sd > 0.05 {
            "*"
        } else {
            ""
        };
        format!(
            "{:<8} [{},{}]{}",
            format!("n = {},", self.n),
            self.scale_min,
            self.scale_max,
            star
        )
    }
}

#[rustfmt::skip]
pub const CASES: &[Case] = &[
    Case { mean: 4.0, sd: 1.50, n: 10,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 3.0, sd: 1.00, n: 12,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 4.0, sd: 2.00, n: 15,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 3.0, sd: 1.20, n: 20,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 4.0, sd: 1.50, n: 20,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 3.5, sd: 1.20, n: 20,  scale_min: 1, scale_max: 7, re_mean: 0.10, re_sd: 0.10 },
    Case { mean: 3.0, sd: 1.00, n: 25,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 4.0, sd: 1.80, n: 25,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 3.5, sd: 1.00, n: 30,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 4.0, sd: 1.50, n: 30,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 3.0, sd: 1.00, n: 40,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 4.0, sd: 1.50, n: 40,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 4.0, sd: 1.50, n: 50,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 3.0, sd: 1.00, n: 50,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 4.0, sd: 1.80, n: 60,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 3.0, sd: 1.00, n: 60,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 4.0, sd: 1.50, n: 75,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 3.0, sd: 1.00, n: 75,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 4.0, sd: 1.50, n: 100, scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05 },
    Case { mean: 3.0, sd: 1.00, n: 100, scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05 },
];

#[test]
fn labels_line_up() {
    assert_eq!(CASES[0].label(), "n = 10,  [1,7]");
    assert_eq!(CASES[5].label(), "n = 20,  [1,7]*");
    assert_eq!(CASES[19].label(), "n = 100, [1,5]");
}
