//! Shared benchmark infrastructure: the `Case` struct and the canonical case list.

pub struct Case {
    pub mean: f64,
    pub sd: f64,
    pub n: i32,
    pub scale_min: i32,
    pub scale_max: i32,
    pub re_mean: f64,
    pub re_sd: f64,
    pub label: &'static str,
}

#[rustfmt::skip]
pub const CASES: &[Case] = &[
    Case { mean: 4.0, sd: 1.50, n: 10,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=10,  [1,7]"  },
    Case { mean: 3.0, sd: 1.00, n: 12,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=12,  [1,5]"  },
    Case { mean: 4.0, sd: 2.00, n: 15,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=15,  [1,7]"  },
    Case { mean: 3.0, sd: 1.20, n: 20,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=20,  [1,5]"  },
    Case { mean: 4.0, sd: 1.50, n: 20,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=20,  [1,7]"  },
    Case { mean: 3.5, sd: 1.20, n: 20,  scale_min: 1, scale_max: 7, re_mean: 0.10, re_sd: 0.10, label: "n=20,  [1,7]*" },
    Case { mean: 3.0, sd: 1.00, n: 25,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=25,  [1,5]"  },
    Case { mean: 4.0, sd: 1.80, n: 25,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=25,  [1,7]"  },
    Case { mean: 3.5, sd: 1.00, n: 30,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=30,  [1,5]"  },
    Case { mean: 4.0, sd: 1.50, n: 30,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=30,  [1,7]"  },
    Case { mean: 3.0, sd: 1.00, n: 40,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=40,  [1,5]"  },
    Case { mean: 4.0, sd: 1.50, n: 40,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=40,  [1,7]"  },
    Case { mean: 4.0, sd: 1.50, n: 50,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=50,  [1,7]"  },
    Case { mean: 3.0, sd: 1.00, n: 50,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=50,  [1,5]"  },
    Case { mean: 4.0, sd: 1.80, n: 60,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=60,  [1,7]"  },
    Case { mean: 3.0, sd: 1.00, n: 60,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=60,  [1,5]"  },
    Case { mean: 4.0, sd: 1.50, n: 75,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=75,  [1,7]"  },
    Case { mean: 3.0, sd: 1.00, n: 75,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=75,  [1,5]"  },
    Case { mean: 4.0, sd: 1.50, n: 100, scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=100, [1,7]"  },
    Case { mean: 3.0, sd: 1.00, n: 100, scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=100, [1,5]"  },
];
