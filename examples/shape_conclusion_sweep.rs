//! Which shape conclusions does CLOSURE reach that SD-only checks do not?
//! The numbers behind `protocols/when-closure-is-useful.md`.
//!
//! Run: cargo run --release --example shape_conclusion_sweep

fn main() {
    let r = 0.005; // rounding error of a two-decimal mean and SD
    for (hi, n) in [(5, 20), (5, 50), (7, 30)] {
        let third = (hi as f64 - 1.0) / 3.0;
        let mut t = [0u32; 12];
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
                // The mean-free single-peak ceiling on the population SD,
                // (2m + 1)/6 for m = hi - 1, against the lowest admissible SD.
                let var_lo = (sd - r).powi(2) * (n as f64 - 1.0) / n as f64;
                let ceiling = var_lo > ((2.0 * hi as f64 - 1.0) / 6.0).powi(2) + 1e-9;
                // The library's flag: Navard et al.'s B_mu at every
                // admissible mean, in exact integers.
                let navard = res.metrics_horns.beyond_single_peak;
                let dip = s.deficit_min > 0;
                let multi = s.must_be(|c| c.is_multimodal()) == Some(true);
                let no_bell = s.can_be_bell_shaped() == Some(false);
                let middle = (mean - 1.0) >= third && (mean - 1.0) <= 2.0 * third;
                for (i, hit) in [
                    true,
                    multi,
                    multi && ceiling,
                    multi && navard,
                    ceiling,
                    navard,
                    dip,
                    dip && ceiling,
                    dip && navard,
                    no_bell,
                    middle,
                    middle && no_bell && !navard,
                ]
                .into_iter()
                .enumerate()
                {
                    t[i] += hit as u32;
                }
            }
        }
        println!(
            "1..{hi} n={n}: consistent {}; must be multimodal {} (ceiling catches {}, \
             B_mu flag catches {}); mean-free ceiling {}; B_mu flag {}; every sample \
             has a dip {} (ceiling catches {}, B_mu flag catches {}); cannot be a bell \
             {}; middle-third means {}, of which no bell and no flag {}",
            t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11]
        );
    }
}
