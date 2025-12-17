//! Unified trait interface for distribution-finding algorithms (CLOSURE and SPRITE)
//!
//! This module provides a consistent API for both CLOSURE and SPRITE algorithms
//! through the DistributionFinder trait, enabling polymorphic usage and ensuring
//! API homogeneity across different distribution reconstruction methods.

use crate::sprite::sprite_parallel;
use crate::sprite_types::RestrictionsOption;
use crate::{
    dfs_parallel, dfs_parallel_streaming, empty_result_list, sprite::sprite_parallel_streaming,
    FloatType, IntegerType, ParquetConfig, ResultListFromMeanSdN, StreamingConfig, StreamingResult,
};
use std::collections::HashMap;

/// Configuration for SPRITE-specific parameters
///
/// This struct encapsulates the algorithm-specific parameters that SPRITE requires
/// beyond the common distribution-finding parameters. These control the randomized
/// search behavior and constraints on the generated distributions.
#[derive(Debug, Clone)]
pub struct SpriteConfig {
    /// Number of items (responses) in the questionnaire
    pub n_items: u32,
    /// Exact count constraints for specific values
    /// Maps value -> required count in the distribution
    pub restrictions_exact: Option<HashMap<i32, usize>>,
    /// Minimum occurrence constraints
    pub restrictions_minimum: RestrictionsOption,
    /// Skip statistical testing of generated distributions
    pub dont_test: bool,
}

impl Default for SpriteConfig {
    fn default() -> Self {
        Self {
            n_items: 1,
            restrictions_exact: None,
            restrictions_minimum: RestrictionsOption::Default,
            dont_test: false,
        }
    }
}

/// Unified trait for distribution-finding algorithms
///
/// This trait provides a consistent interface for algorithms that reconstruct
/// possible raw data distributions from summary statistics (mean, SD, N).
/// Both CLOSURE (exhaustive search) and SPRITE (stochastic search) implement
/// this trait, enabling polymorphic usage.
///
/// # Type Parameters
/// - `T`: Floating-point type for mean, SD, and tolerances (e.g., f32, f64)
/// - `U`: Integer type for sample values and counts (e.g., i32, i64, u32)
/// - `Config`: Algorithm-specific configuration (unit type () for CLOSURE, SpriteConfig for SPRITE)
///
/// # Example
/// ```ignore
/// // Using CLOSURE (no algorithm-specific config needed)
/// let results = ClosureFinder::find_distributions(
///     2.5, 1.0, 20, 1, 5,
///     0.05, 0.05,
///     (),  // No extra config for CLOSURE
///     None, None,
/// );
///
/// // Using SPRITE (requires SpriteConfig and RNG)
/// let config = SpriteConfig::default();
/// let results = SpriteFinder::find_distributions(
///     2.5, 1.0, 20, 1, 5,
///     0.05, 0.05,
///     config,
///     None, None,
/// );
/// ```
pub trait DistributionFinder<T, U, Config = ()>
where
    T: FloatType,
    U: IntegerType + 'static,
{
    /// Find distributions in memory with full statistics
    ///
    /// Generates all valid distributions (or up to `stop_after`) that match
    /// the given summary statistics within the specified tolerances.
    /// Returns complete results in memory with statistical summaries.
    ///
    /// # Parameters
    /// - `mean`: Target mean value
    /// - `sd`: Target standard deviation
    /// - `n`: Sample size (number of values in each distribution)
    /// - `scale_min`: Minimum value in the distribution scale
    /// - `scale_max`: Maximum value in the distribution scale
    /// - `rounding_error_mean`: Tolerance for mean (e.g., 0.05 for ±0.05)
    /// - `rounding_error_sd`: Tolerance for SD (e.g., 0.05 for ±0.05)
    /// - `config`: Algorithm-specific configuration
    /// - `parquet_config`: Optional Parquet file output configuration
    /// - `stop_after`: Optional limit on number of distributions to find
    ///
    /// # Returns
    /// `ResultListFromMeanSdN<U>` containing all distributions and statistics
    fn find_distributions(
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        config: Config,
        parquet_config: Option<ParquetConfig>,
        stop_after: Option<usize>,
    ) -> ResultListFromMeanSdN<U>;

    /// Find distributions in streaming mode for memory efficiency
    ///
    /// Generates distributions and streams them directly to Parquet files
    /// without holding all results in memory. Suitable for large result sets.
    ///
    /// # Parameters
    /// Same as `find_distributions`, except:
    /// - `streaming_config`: Required configuration for streaming output files
    ///
    /// # Returns
    /// `StreamingResult` containing the total count and file path
    fn find_distributions_streaming(
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        config: Config,
        streaming_config: StreamingConfig,
        stop_after: Option<usize>,
    ) -> StreamingResult;

    /// Returns the name of the algorithm
    ///
    /// # Returns
    /// A static string identifying the algorithm (e.g., "CLOSURE", "SPRITE")
    fn algorithm_name() -> &'static str;
}

/// CLOSURE algorithm implementation (exhaustive search)
///
/// Zero-sized type that implements DistributionFinder for the CLOSURE algorithm.
/// CLOSURE performs exhaustive depth-first search to find all valid distributions.
///
/// # Example
/// ```ignore
/// let results = ClosureFinder::find_distributions(
///     2.5, 1.0, 20, 1, 5,
///     0.05, 0.05,
///     (),  // No config needed
///     None, None,
/// );
/// println!("Algorithm: {}", ClosureFinder::algorithm_name());
/// ```
pub struct ClosureFinder;

impl<T, U> DistributionFinder<T, U, ()> for ClosureFinder
where
    T: FloatType,
    U: IntegerType + 'static,
{
    #[inline]
    fn find_distributions(
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        _config: (),
        parquet_config: Option<ParquetConfig>,
        stop_after: Option<usize>,
    ) -> ResultListFromMeanSdN<U> {
        dfs_parallel(
            mean,
            sd,
            n,
            scale_min,
            scale_max,
            rounding_error_mean,
            rounding_error_sd,
            parquet_config,
            stop_after,
        )
    }

    #[inline]
    fn find_distributions_streaming(
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        _config: (),
        streaming_config: StreamingConfig,
        stop_after: Option<usize>,
    ) -> StreamingResult {
        dfs_parallel_streaming(
            mean,
            sd,
            n,
            scale_min,
            scale_max,
            rounding_error_mean,
            rounding_error_sd,
            streaming_config,
            stop_after,
        )
    }

    fn algorithm_name() -> &'static str {
        "CLOSURE"
    }
}

/// SPRITE algorithm implementation (stochastic search)
///
/// Zero-sized type that implements DistributionFinder for the SPRITE algorithm.
/// SPRITE uses randomized search to efficiently find valid distributions.
///
/// Note: The trait implementation creates its own thread-local RNG for convenience.
/// If you need full control over the RNG (e.g., for reproducibility with a specific seed),
/// use the `sprite_parallel` and `sprite_parallel_streaming` functions directly.
///
/// # Example
/// ```ignore
/// let config = SpriteConfig {
///     n_items: 1,
///     restrictions_exact: None,
///     restrictions_minimum: RestrictionsOption::Default,
///     dont_test: false,
/// };
///
/// let results = SpriteFinder::find_distributions(
///     2.5, 1.0, 20, 1, 5,
///     0.05, 0.05,
///     config,
///     None, None,
/// );
/// ```
pub struct SpriteFinder;

impl<T, U> DistributionFinder<T, U, SpriteConfig> for SpriteFinder
where
    T: FloatType,
    U: IntegerType + 'static,
{
    #[inline]
    fn find_distributions(
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        config: SpriteConfig,
        parquet_config: Option<ParquetConfig>,
        stop_after: Option<usize>,
    ) -> ResultListFromMeanSdN<U> {
        // Create a thread-local RNG for trait compatibility
        let mut rng = rand::rng();

        sprite_parallel(
            mean,
            sd,
            n,
            scale_min,
            scale_max,
            rounding_error_mean,
            rounding_error_sd,
            config.n_items,
            config.restrictions_exact,
            config.restrictions_minimum,
            config.dont_test,
            parquet_config,
            stop_after,
            &mut rng,
        )
        .unwrap_or_else(|_| empty_result_list(scale_min, scale_max))
    }

    #[inline]
    fn find_distributions_streaming(
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        config: SpriteConfig,
        streaming_config: StreamingConfig,
        stop_after: Option<usize>,
    ) -> StreamingResult {
        sprite_parallel_streaming(
            mean,
            sd,
            n,
            scale_min,
            scale_max,
            rounding_error_mean,
            rounding_error_sd,
            config.n_items,
            config.restrictions_exact,
            config.restrictions_minimum,
            config.dont_test,
            streaming_config,
            stop_after,
        )
    }

    fn algorithm_name() -> &'static str {
        "SPRITE"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closure_finder_trait() {
        // Test CLOSURE through the trait
        let results = ClosureFinder::find_distributions(
            3.0_f64, // mean
            1.0_f64, // sd
            5_i32,   // n
            1_i32,   // scale_min
            5_i32,   // scale_max
            0.05,    // rounding_error_mean
            0.05,    // rounding_error_sd
            (),      // no config
            None,    // no parquet
            Some(10), // stop after 10
        );

        assert!(!results.results.sample.is_empty());
        assert!(results.results.sample.len() <= 10);
        assert_eq!(
            <ClosureFinder as DistributionFinder<f64, i32, ()>>::algorithm_name(),
            "CLOSURE"
        );
    }

    #[test]
    fn test_sprite_finder_trait() {
        // Test SPRITE through the trait
        let config = SpriteConfig::default();

        let results = SpriteFinder::find_distributions(
            2.2_f64, // mean
            1.3_f64, // sd
            20_i32,  // n
            1_i32,   // scale_min
            5_i32,   // scale_max
            0.05,    // rounding_error_mean
            0.05,    // rounding_error_sd
            config,
            None,    // no parquet
            Some(5), // stop after 5
        );

        assert!(!results.results.sample.is_empty());
        assert!(results.results.sample.len() <= 5);
        assert_eq!(
            <SpriteFinder as DistributionFinder<f64, i32, SpriteConfig>>::algorithm_name(),
            "SPRITE"
        );
    }

    #[test]
    fn test_sprite_config_default() {
        let config = SpriteConfig::default();
        assert_eq!(config.n_items, 1);
        assert!(config.restrictions_exact.is_none());
        assert!(!config.dont_test);
    }

    #[test]
    fn test_closure_streaming() {
        let config = StreamingConfig {
            file_path: "test_trait_closure/".to_string(),
            batch_size: 100,
            show_progress: false,
        };

        let _ = std::fs::create_dir("test_trait_closure");

        let result = ClosureFinder::find_distributions_streaming(
            3.0_f64, // mean
            1.0_f64, // sd
            5_i32,   // n
            1_i32,   // scale_min
            5_i32,   // scale_max
            0.05,    // rounding_error_mean
            0.05,    // rounding_error_sd
            (),      // no config
            config,
            Some(5), // stop after 5
        );

        assert!(result.total_combinations > 0);

        // Clean up
        let _ = std::fs::remove_file("test_trait_closure/samples.parquet");
        let _ = std::fs::remove_file("test_trait_closure/horns.parquet");
        let _ = std::fs::remove_file("test_trait_closure/metrics_main.parquet");
        let _ = std::fs::remove_file("test_trait_closure/metrics_horns.parquet");
        let _ = std::fs::remove_file("test_trait_closure/frequency.parquet");
        let _ = std::fs::remove_dir("test_trait_closure");
    }

    #[test]
    fn test_sprite_streaming() {
        let sprite_config = SpriteConfig::default();
        let streaming_config = StreamingConfig {
            file_path: "test_trait_sprite/".to_string(),
            batch_size: 100,
            show_progress: false,
        };

        let _ = std::fs::create_dir("test_trait_sprite");

        let result = SpriteFinder::find_distributions_streaming(
            2.2_f64,       // mean
            1.3_f64,       // sd
            20_i32,        // n
            1_i32,         // scale_min
            5_i32,         // scale_max
            0.05,          // rounding_error_mean
            0.05,          // rounding_error_sd
            sprite_config,
            streaming_config,
            Some(5), // stop after 5
        );

        assert!(result.total_combinations > 0);

        // Clean up
        let _ = std::fs::remove_file("test_trait_sprite/samples.parquet");
        let _ = std::fs::remove_file("test_trait_sprite/horns.parquet");
        let _ = std::fs::remove_file("test_trait_sprite/metrics_main.parquet");
        let _ = std::fs::remove_file("test_trait_sprite/metrics_horns.parquet");
        let _ = std::fs::remove_file("test_trait_sprite/frequency.parquet");
        let _ = std::fs::remove_dir("test_trait_sprite");
    }
}
