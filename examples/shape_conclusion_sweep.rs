//! Which shape conclusions does CLOSURE reach that SD-only checks do not?
//! The numbers behind `protocols/when-closure-is-useful.md`.
//!
//! Run: cargo run --release --example shape_conclusion_sweep

/// Navard, Seaman & Young (1993), Corollary 4.1.2: variance bound for a
/// unimodal distribution on categories 1..n with mean mu.
fn b_mu(mu: f64, n: f64) -> f64 {
    if 2.0 * mu > n + 1.0 {
        ((4.0 * n + 1.0) * mu - n * (n + 1.0)) / 3.0 - mu * mu
    } else if 2.0 * mu == n + 1.0 {
        (n * n - 1.0) / 12.0
    } else {
        ((2.0 * n + 5.0) * mu - 2.0 * (n + 1.0)) / 3.0 - mu * mu
    }
}

fn main() {
    let r = 0.005; // rounding error of a two-decimal mean and SD
    for (hi, n) in [(5, 20), (5, 50), (7, 30)] {
        let third = (hi as f64 - 1.0) / 3.0;
        let mut t = [0u32; 9];
        for mean_x10 in 12..=(hi * 10 - 2) {
            for sd_x10 in 3..=26 {
                let (mean, sd) = (mean_x10 as f64 / 10.0, sd_x10 as f64 / 10.0);
                let res = closure_core::closure_parallel::<f64, i32>(
                    mean, sd, n, 1, hi, r, r, 1, None, None, None,
                )
                .unwrap();
                if res.results.is_empty() {
                    continue; // inconsistent: no shape to infer
                }
                let s = &res.modality_shapes;
                let ceiling = res.metrics_horns.beyond_single_peak;
                let var_lo = (sd - r).powi(2) * (n as f64 - 1.0) / n as f64;
                let b = (0..=100)
                    .map(|i| b_mu(mean - r + 2.0 * r * i as f64 / 100.0, hi as f64))
                    .fold(f64::MIN, f64::max);
                let navard = var_lo > b + 1e-9;
                let dip = s.deficit_min > 0;
                let multi = s.must_be(|c| c.is_multimodal()) == Some(true);
                let no_bell = s.can_be_bell_shaped() == Some(false);
                let middle = (mean - 1.0) >= third && (mean - 1.0) <= 2.0 * third;
                for (i, hit) in [
                    true,
                    multi,
                    ceiling,
                    dip,
                    dip && ceiling,
                    dip && navard,
                    no_bell,
                    middle,
                    middle && no_bell && !ceiling,
                ]
                .into_iter()
                .enumerate()
                {
                    t[i] += hit as u32;
                }
            }
        }
        println!(
            "1..{hi} n={n}: consistent {}; must be multimodal {}; ceiling flag {}; \
             every sample has a dip {} (ceiling catches {}, B_mu catches {}); \
             cannot be a bell {}; middle-third means {}, of which no bell and no flag {}",
            t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]
        );
    }
}
