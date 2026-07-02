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

# Golden pack C.2 3-study case: y=[.1,.3,.5], v=[.02,.03,.05], k=3.
# Exact fractions: theta_F=15/62, Q=76/31, tau^2=14/2000=0.007, I^2=7/38.
dl <- dersimonian_laird(c(0.1, 0.3, 0.5), c(0.02, 0.03, 0.05))
assert_close(dl$theta_fixed, 15 / 62,             1e-12, "DL:theta_fixed=15/62")
assert_close(dl$Q,           76 / 31,             1e-12, "DL:Q=76/31")
assert_close(dl$tau2,        0.007,               1e-12, "DL:tau2=0.007")
assert_close(dl$I2,          7 / 38,              1e-12, "DL:I2=7/38")
assert_close(dl$sum_wstar_y, 20.58374163637322,   1e-9,  "DL:sum(w*.y)")
assert_close(dl$sum_wstar,   81.60792371318689,   1e-9,  "DL:sum(w*)")
assert_close(dl$pooled,      0.2522272433828276,  1e-12, "DL:pooled")
assert_close(dl$se,          0.11069648626900815, 1e-9,  "DL:SE")
cat(sprintf("PASS (1) DerSimonian-Laird: Q=%.15g tau2=%.15g I2=%.15g pooled=%.16g\n",
            dl$Q, dl$tau2, dl$I2, dl$pooled))

## ---------------------------------------------------------------------------
## (2) Schuirmann's TOST for two means (Welch-Satterthwaite df)
##     Cross-checks src/membench/stats/equivalence.py; golden pack C.1.
##     Two one-sided tests against an equivalence margin +/- delta:
##       diff     = mean_a - mean_b
##       SE       = sqrt(sd_a^2/n_a + sd_b^2/n_b)
##       df       = SE^4 / ( (sd_a^2/n_a)^2/(n_a-1) + (sd_b^2/n_b)^2/(n_b-1) )
##       t_lower  = (diff + delta) / SE ; p_lower = P(T_df >  t_lower)
##       t_upper  = (diff - delta) / SE ; p_upper = P(T_df <  t_upper)
##       p_value  = max(p_lower, p_upper); equivalent iff p_value < alpha
##     Equivalence is concluded only when BOTH one-sided tails reject, i.e.
##     the larger tail probability is below alpha.
## ---------------------------------------------------------------------------
tost_two_means <- function(mean_a, sd_a, n_a, mean_b, sd_b, n_b, delta) {
  va <- sd_a^2 / n_a
  vb <- sd_b^2 / n_b
  diff <- mean_a - mean_b
  se <- sqrt(va + vb)
  df <- (va + vb)^2 / (va^2 / (n_a - 1) + vb^2 / (n_b - 1))
  t_lower <- (diff + delta) / se
  t_upper <- (diff - delta) / se
  p_lower <- pt(t_lower, df, lower.tail = FALSE)
  p_upper <- pt(t_upper, df, lower.tail = TRUE)
  list(diff = diff, se = se, df = df,
       t_lower = t_lower, t_upper = t_upper,
       p_lower = p_lower, p_upper = p_upper,
       p_value = max(p_lower, p_upper))
}

# Golden pack C.1: a=(10.0, sd 2, n 50), b=(10.5, sd 2, n 50), delta=1, alpha=0.05.
# diff=-0.5, SE=0.40, df collapses to 2(n-1)=98; t_lower=1.25, t_upper=-3.75.
# p_value=P(T98>1.25)=0.10713816216177832 > 0.05 => NOT equivalent (correct verdict).
tost <- tost_two_means(10.0, 2.0, 50, 10.5, 2.0, 50, delta = 1.0)
assert_close(tost$diff,    -0.5,                    1e-12, "TOST:diff")
assert_close(tost$se,       0.40,                   1e-12, "TOST:SE")
assert_close(tost$df,       98,                     1e-9,  "TOST:df=98")
assert_close(tost$t_lower,  1.25,                   1e-12, "TOST:t_lower=1.25")
assert_close(tost$t_upper, -3.75,                   1e-12, "TOST:t_upper=-3.75")
assert_close(tost$p_lower,  0.10713816216177832,    1e-12, "TOST:p_lower")
assert_close(tost$p_upper,  0.00014966157362142826, 1e-12, "TOST:p_upper")
assert_close(tost$p_value,  0.10713816216177832,    1e-12, "TOST:p_value")
assert_true(tost$p_value >= 0.05, "TOST:not-equivalent (p>=alpha)")
cat(sprintf("PASS (2) Schuirmann TOST: t_lower=%.2f t_upper=%.2f df=%g p=%.17g (not equivalent)\n",
            tost$t_lower, tost$t_upper, tost$df, tost$p_value))

## ---------------------------------------------------------------------------
## (3) PAVA isotonic regression (pool-adjacent-violators)
##     Cross-checks src/membench/stats/isotonic.py; golden pack C.3.
##     Non-decreasing L2 fit: sweep left-to-right, keep a stack of blocks
##     (weighted mean + weight + span); whenever the incoming block violates
##     monotonicity against its predecessor, pool them (weighted average) and
##     repeat. Preserves the weighted sum; idempotent. A non-increasing fit is
##     the mirror image: reverse, fit non-decreasing, reverse back.
## ---------------------------------------------------------------------------
pava_increasing <- function(y, w = rep(1, length(y))) {
  vals <- as.numeric(y)
  wts <- as.numeric(w)
  stopifnot(length(vals) == length(wts), all(wts > 0))
  bv <- numeric(0)  # block value (weighted mean)
  bw <- numeric(0)  # block weight
  bc <- integer(0)  # block span (# original points)
  for (i in seq_along(vals)) {
    bv <- c(bv, vals[i]); bw <- c(bw, wts[i]); bc <- c(bc, 1L)
    while (length(bv) > 1L && bv[length(bv) - 1L] > bv[length(bv)]) {
      m <- length(bv)
      new_w <- bw[m - 1L] + bw[m]
      new_v <- (bv[m - 1L] * bw[m - 1L] + bv[m] * bw[m]) / new_w
      new_c <- bc[m - 1L] + bc[m]
      keep <- seq_len(m - 2L)
      bv <- c(bv[keep], new_v); bw <- c(bw[keep], new_w); bc <- c(bc[keep], new_c)
    }
  }
  rep(bv, bc)
}

pava_decreasing <- function(y, w = rep(1, length(y))) {
  rev(pava_increasing(rev(y), rev(w)))
}

# Golden pack C.3: non-decreasing fit of [1,2,3,2,1] -> [1,2,2,2,2]
#   (blocks [1],[2],[3]; add 2: 3>2 -> pool 2.5; add 1: 2.5>1 -> pool 2.0).
# Non-increasing fit (mirror) -> [2,2,2,2,1].
iso_in <- c(1, 2, 3, 2, 1)
iso_up <- pava_increasing(iso_in)
iso_dn <- pava_decreasing(iso_in)
assert_true(iso_up == c(1, 2, 2, 2, 2), "PAVA:increasing[1,2,2,2,2]")
assert_true(iso_dn == c(2, 2, 2, 2, 1), "PAVA:decreasing[2,2,2,2,1]")
# Sum-preserving invariant (both fits keep the total).
assert_close(sum(iso_up), sum(iso_in), 1e-12, "PAVA:sum-preserved-up")
assert_close(sum(iso_dn), sum(iso_in), 1e-12, "PAVA:sum-preserved-dn")
# Idempotence: refitting an already-isotonic sequence is a no-op.
assert_true(pava_increasing(iso_up) == iso_up, "PAVA:idempotent")
# Weighted pooling: values [2,0], weights [1,3] -> pooled mean (2*1+0*3)/4=0.5.
iso_w <- pava_increasing(c(2, 0), c(1, 3))
assert_true(iso_w == c(0.5, 0.5), "PAVA:weighted[0.5,0.5]")
# R^2_iso on [1,2,3,2,1] (increasing): mean 1.8, SS_total 2.8, SS_resid 2.0 -> 2/7.
iso_mu <- mean(iso_in)
ss_total <- sum((iso_in - iso_mu)^2)
ss_resid <- sum((iso_in - iso_up)^2)
r2_iso <- 1 - ss_resid / ss_total
assert_close(ss_total, 2.8, 1e-12, "PAVA:SS_total=2.8")
assert_close(ss_resid, 2.0, 1e-12, "PAVA:SS_resid=2.0")
assert_close(r2_iso, 2 / 7, 1e-12, "PAVA:R2_iso=2/7")
cat(sprintf("PASS (3) PAVA isotonic: up=[%s] dn=[%s] R2_iso=%.16g\n",
            paste(iso_up, collapse = ","), paste(iso_dn, collapse = ","), r2_iso))

## ---------------------------------------------------------------------------
## (4) Wilson score interval + Cohen's h
##     Cross-checks src/membench/metrics/factorization.py::wilson_interval and
##     src/membench/stats/effect_size.py; golden pack C.6 and C.7.
##     Wilson score CI (clamped to [0,1]):
##       center = (p_hat + z^2/2n) / (1 + z^2/n)
##       margin = z*sqrt(p_hat(1-p_hat)/n + z^2/4n^2) / (1 + z^2/n)
##       z = qnorm(1 - (1-conf)/2)  (95% -> 1.959963984540054)
##     Cohen's h (arcsine-transformed proportion effect size):
##       h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2))
## ---------------------------------------------------------------------------
wilson_interval <- function(successes, trials, conf = 0.95) {
  z <- qnorm(1 - (1 - conf) / 2)
  p_hat <- successes / trials
  denom <- 1 + z^2 / trials
  center <- (p_hat + z^2 / (2 * trials)) / denom
  margin <- (z * sqrt(p_hat * (1 - p_hat) / trials + z^2 / (4 * trials^2))) / denom
  list(low = max(0, center - margin), high = min(1, center + margin),
       center = center, margin = margin)
}

cohen_h <- function(p1, p2) 2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2))
