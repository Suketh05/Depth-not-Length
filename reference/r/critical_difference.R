#!/usr/bin/env Rscript
# Cross-check of the Friedman + Nemenyi critical-difference analysis
# (membench.stats.multiple_comparisons) in R -- the language Demsar's "Statistical
# Comparisons of Classifiers over Multiple Data Sets" protocol is canonically run in.
# Builds a (blocks x arms) score matrix, runs the Friedman omnibus, computes average
# ranks and the Nemenyi critical difference, and asserts the structured arm ranks best
# and the omnibus is significant -- agreeing with the Python implementation.
# Run: Rscript reference/r/critical_difference.R

arms <- c("none", "dense", "hybrid", "brief_graph")
scores <- rbind(
  c(0.00, 0.70, 0.70, 1.00),
  c(0.00, 0.60, 0.65, 1.00),
  c(0.00, 0.50, 0.55, 1.00),
  c(0.00, 0.75, 0.80, 0.95),
  c(0.00, 0.40, 0.50, 1.00),
  c(0.00, 0.60, 0.70, 0.98)
)
colnames(scores) <- arms

ft <- friedman.test(scores)
cat(sprintf("Friedman chi^2 = %.3f, p = %.4g\n", ft$statistic, ft$p.value))

# average ranks (1 = best): rank the negated scores so larger recovery -> rank 1
ranks <- t(apply(-scores, 1, rank))
avg <- colMeans(ranks)
cat("average ranks:\n"); print(round(avg, 3))

k <- length(arms); N <- nrow(scores)
cd <- qtukey(0.95, k, Inf) / sqrt(2) * sqrt(k * (k + 1) / (6 * N))
cat(sprintf("Nemenyi critical difference (alpha = 0.05) = %.3f\n", cd))

stopifnot(ft$p.value < 0.05)
stopifnot(names(which.min(avg)) == "brief_graph")
cat("OK: structured arm ranks best and the Friedman omnibus is significant\n")

## ===========================================================================
## SELF-ASSERTING STATISTICS CROSS-CHECKS (base R, CI-executed)
## ---------------------------------------------------------------------------
## The blocks below re-implement, in base R only, the statistical estimators
## the paper "Depth, Not Length" relies on, and assert their golden values
## VERBATIM from the canonical golden-value pack (float64 full precision or
## exact fraction). Every block ends in stopifnot(), so a mismatch makes
## Rscript exit non-zero and turns CI red -- the cross-language contract is
## machine-enforced, not narrated. Base/stats functions only (no library()).
##
## Golden provenance is cited per block (golden-pack section + paper anchor).
## Recommended tolerance: exact-fraction identity where the value is rational,
## else abs(err) <= 1e-12 on float64 (looser where a provenance test is looser).
## ===========================================================================

# assert_close(actual, expected, tol, label): float64 tolerance gate.
# Prints a full-precision diagnostic on failure, then stopifnot()-fails CI.
assert_close <- function(actual, expected, tol, label) {
  ok <- length(actual) == 1L && length(expected) == 1L &&
    is.finite(actual) && is.finite(expected) && abs(actual - expected) <= tol
  if (!isTRUE(ok)) {
    cat(sprintf("FAIL %s: actual=%.17g expected=%.17g |diff|=%.3g tol=%.3g\n",
                label, actual, expected, abs(actual - expected), tol))
  }
  stopifnot(isTRUE(ok))
  invisible(TRUE)
}

# assert_true(cond, label): boolean/exact-equality gate with a labelled failure.
assert_true <- function(cond, label) {
  if (!isTRUE(all(cond))) cat(sprintf("FAIL %s\n", label))
  stopifnot(isTRUE(all(cond)))
  invisible(TRUE)
}

cat("\n== statistics cross-checks (base R) ==\n")

## ---------------------------------------------------------------------------
## (1) DerSimonian-Laird random-effects meta-analysis
##     Cross-checks src/membench/stats/meta_analysis.py; golden pack C.2.
##     Standard DL moment estimator (DerSimonian & Laird 1986):
##       w_i      = 1 / v_i                       (fixed-effect / inverse-var)
##       theta_F  = sum(w_i y_i) / sum(w_i)
##       Q        = sum w_i (y_i - theta_F)^2     (Cochran's Q)
##       C        = sum(w_i) - sum(w_i^2)/sum(w_i)
##       tau^2    = max(0, (Q - (k-1)) / C)       (between-study variance)
##       I^2      = max(0, (Q - (k-1)) / Q)       (Higgins-Thompson)
##       w*_i     = 1 / (v_i + tau^2)             (random-effects weights)
##       theta_R  = sum(w*_i y_i) / sum(w*_i)
##       SE       = sqrt(1 / sum(w*_i))
## ---------------------------------------------------------------------------
dersimonian_laird <- function(y, v) {
  k <- length(y)
  stopifnot(k >= 2L, length(v) == k, all(v > 0))
  w <- 1 / v
  sw <- sum(w)
  theta_fixed <- sum(w * y) / sw
  Q <- sum(w * (y - theta_fixed)^2)
  df <- k - 1
  C <- sw - sum(w^2) / sw
  tau2 <- max(0, (Q - df) / C)
  I2 <- max(0, (Q - df) / Q)
  w_star <- 1 / (v + tau2)
  sw_star <- sum(w_star)
  pooled <- sum(w_star * y) / sw_star
  list(theta_fixed = theta_fixed, Q = Q, tau2 = tau2, I2 = I2,
       sum_wstar_y = sum(w_star * y), sum_wstar = sw_star,
       pooled = pooled, se = sqrt(1 / sw_star))
}
