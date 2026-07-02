// Tests for the depth-theory Monte-Carlo simulator.
//
// Golden values are asserted VERBATIM from the canonical golden pack
// (theory section A.1 "Similarity recovery" and A.3 "Gap and crossover"), which
// transcribes the paper's synthetic operating point (sec:depth) and the closed
// forms of eq:simceiling / thm:struct / eq:gd / prop:cross. Every literal below
// was produced by the Python reference implementation, never by running this Go
// code. Go's math.Pow differs from the reference (libm) by at most ~1 ULP
// (~1e-17), far inside the 1e-12 tolerance the golden pack permits for
// non-exact-fraction float64 goldens.
package depthsim

import (
	"math"
	"testing"
)

// analyticTol is the golden-pack cross-language tolerance for float64 values
// that are not asserted as exact fractions (golden pack header + note ULP).
const analyticTol = 1e-12

// Golden P_ret values under PaperCalibration (s0=0.70, rho=0.67, q=0.97),
// golden pack A.1 table, d = 1..5. Similarity = s0^d * rho^(d(d+1)/2);
// Structured = q^d.
var (
	goldenSimilarity = map[int]float64{
		1: 0.469,
		2: 0.14737387000000002,
		3: 0.031027225083967002,
		4: 0.004376633568728781,
		5: 0.0004136302005695817,
	}
	goldenStructured = map[int]float64{
		1: 0.97,
		2: 0.9409,
		3: 0.912673,
		4: 0.8852928099999999,
		5: 0.8587340256999999,
	}
	// g(d) = q^d - P_sim(d), golden pack A.1 "g(d)" column.
	goldenGap = map[int]float64{
		1: 0.501,
		2: 0.7935261299999999,
		3: 0.8816457749160329,
		4: 0.8809161764312712,
		5: 0.8583203954994303,
	}
)

func closeTo(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("%s: got %.17g, want %.17g (|diff|=%.3g > tol %.3g)", name, got, want, math.Abs(got-want), tol)
	}
}

func TestAnalyticPRetMatchesGoldenPack(t *testing.T) {
	m := PaperCalibration
	for d := 1; d <= 5; d++ {
		closeTo(t, "P_sim(d)", m.AnalyticPRet(Similarity, d), goldenSimilarity[d], analyticTol)
		closeTo(t, "P_struct(d)", m.AnalyticPRet(Structured, d), goldenStructured[d], analyticTol)
		closeTo(t, "g(d)", m.Gap(d), goldenGap[d], analyticTol)
	}
}

// TestDepthGapRatio checks the paper's headline "~29x gap" at d=3
// (golden pack A.3): q^3 / P_sim(3) = 29.415231221293272.
func TestDepthGapRatio(t *testing.T) {
	m := PaperCalibration
	ratio := m.AnalyticPRet(Structured, 3) / m.AnalyticPRet(Similarity, 3)
	closeTo(t, "q^3/P_sim(3)", ratio, 29.415231221293272, 1e-9)
}

// simTrials / simSeed / simWorkers are the fixed Monte-Carlo settings shared by
// the empirical tests. The seed is fixed so the (otherwise probabilistic)
// 3-sigma assertion is deterministic and reproducible in CI; N=200k is the size
// the task specifies. seed=7's worst cell lands at ~1.6 sigma, comfortably
// inside the 3-sigma band.
const (
	simTrials  = 200000
	simSeed    = uint64(7)
	simWorkers = 8
)

// TestEmpiricalMatchesAnalyticWithin3SE is the core evidence that the closed-form
// theory (eq:simceiling, thm:struct) describes the generative model: with 200k
// seeded trials the empirical P_ret(d) lands within 3 binomial standard errors of
// the analytic value for BOTH arms at every depth 1..5. The SE is computed here
// as sqrt(p*(1-p)/N) on the analytic p (the estimator's true standard error).
func TestEmpiricalMatchesAnalyticWithin3SE(t *testing.T) {
	m := PaperCalibration
	for d := 1; d <= 5; d++ {
		for _, a := range []Arm{Similarity, Structured} {
			est := m.SimulatePRet(a, d, simTrials, simWorkers, simSeed)
			analytic := m.AnalyticPRet(a, d)
			se := math.Sqrt(analytic * (1 - analytic) / float64(simTrials))
			dev := math.Abs(est.PHat() - analytic)
			if dev > 3*se {
				t.Errorf("%s d=%d: empirical=%.6g analytic=%.6g |dev|=%.3g > 3*SE=%.3g (%.2f sigma)",
					a, d, est.PHat(), analytic, dev, 3*se, dev/se)
			}
		}
	}
}

// TestSchedulingIndependence proves the estimate is a pure function of the
// inputs, not of the goroutine schedule: because each trial is reseeded from its
// GLOBAL index (trialSeed), 1 worker and 8 workers must return the IDENTICAL
// success count (not just statistically close). This is the property that lets
// the concurrent simulator stand in for a serial reference.
func TestSchedulingIndependence(t *testing.T) {
	m := PaperCalibration
	const n = 50000
	for d := 1; d <= 5; d++ {
		for _, a := range []Arm{Similarity, Structured} {
			one := m.SimulatePRet(a, d, n, 1, simSeed)
			eight := m.SimulatePRet(a, d, n, 8, simSeed)
			if one.Successes != eight.Successes {
				t.Fatalf("%s d=%d: 1 worker=%d successes, 8 workers=%d successes (must be identical)",
					a, d, one.Successes, eight.Successes)
			}
			if one.PHat() != eight.PHat() {
				t.Fatalf("%s d=%d: PHat differs across worker counts: %v vs %v", a, d, one.PHat(), eight.PHat())
			}
		}
	}
	// A worker count larger than n (some workers get zero trials) must not change
	// the answer either.
	huge := m.SimulatePRet(Similarity, 3, n, 4096, simSeed)
	base := m.SimulatePRet(Similarity, 3, n, 1, simSeed)
	if huge.Successes != base.Successes {
		t.Fatalf("workers>n changed result: %d vs %d", huge.Successes, base.Successes)
	}
}
