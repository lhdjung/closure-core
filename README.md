# CLOSURE: complete listing of original samples of underlying raw evidence

Implements the novel CLOSURE technique for efficiently reconstructing all possible distributions of raw data from summary statistics. It is not about the Rust feature called closure.

You will likely only need `dfs_parallel()`.

Most of the code was written by Claude 3.5, translating Python code by Nathanael Larigaldie.

## Example

Enter summary data reported in a paper and call `dfs_parallel()`.

```
let min_scale = 1;
let max_scale = 7;
let n = 30;
let target_mean = 5.0;
let target_sum = target_mean * n as f64;
let target_sd = 2.78;
let rounding_error_means = 0.01;
let rounding_error_sums = rounding_error_means * n as f64;
let rounding_error_sds = 0.01;
let output_file = "parallel_results.csv";

dfs_parallel(
    min_scale,
    max_scale,
    n,
    target_sum,
    target_sd,
    rounding_error_sums,
    rounding_error_sds,
    output_file,
)
```

#### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
</sub>