⏺ The mystery is solved. Here's the full chain of what was happening:

  1. src/rust/vendor.tar.xz contains a pre-built vendor directory with all dependencies
  2. The Makevars extracts it on every R build: tar xf rust/vendor.tar.xz
  3. This overwrites src/vendor/closure-core/ with the old version (no empty(), old depth formula)
  4. Our cargo vendor ../vendor updated the extracted directory, but the tarball was still stale
  5. Every devtools::load_all() re-extracted the old tarball, clobbering our fix

  What was fixed:
  - Re-vendored closure-core from the updated lock file (commit 0b84c261 with both empty() and depth = 2)
  - Rebuilt vendor.tar.xz from the updated vendor directory

  You should now be able to run devtools::load_all() successfully. The benchmark should be back to ~30 seconds.
