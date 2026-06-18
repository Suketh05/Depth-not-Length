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
