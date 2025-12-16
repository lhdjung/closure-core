I'm still unsatisfied by the inconsistencies between the CLOSURE and SPRITE APIs. Below are changed I want you to make. Consider whether one or more new traits could help guarantee that the two functions are completely rounded and homogenous, as if made from one piece.

1. The CLOSURE functions, starting with dfs_parallel(), have rounding_error_mean and rounding_error_sd arguments. The SPRITE functions, starting with sprite_parallel(), should have them, too. Remove the m_prec and sd_prec arguments from all functions. That is, SPRITE functions should take the rounding error directly, instead of figuring it out via m_prec and sd_prec.
2. The n_obs: u32 argument should be replaced by a n: U argument, just like in the CLOSURE functions.
3. min_val: i32 should be replaced by scale_min: U and max_val: i32 should be replaced by scale_max: U.
4. The n_items argument should be cut completely.
5. n_distributions: usize should be replaced by stop_after: Option<usize>.
6. The order of arguments in the SPRITE functions should be just like in the CLOSURE functions.

NOTE after refactoring: I interjected during the process to keep n_items because it is more central than the other old SPRITE arguments.