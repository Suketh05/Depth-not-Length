// Package depthsim is a deterministic, concurrent Monte-Carlo simulator for the
// depth theory of retrieval failure in "Depth, Not Length: A Benchmark and
// Theory of Memory, Retrieval, and Decision Compliance in LLM Coding Agents".
//
// It models a depth-d causal retrieval chain as a series system of independent
// per-hop survival events (paper eq:perhop, Assumption ass:hopindep):
//
//	P_ret(d) = Prod_{k=1}^{d} f_k
//
// with two arms:
//
//   - Similarity (Assumption ass:geometric, Thm thm:sim / eq:simceiling): hop k
//     survives with probability s_k = s0 * rho^k, giving the super-geometric
//     ceiling P_ret^sim(d) = s0^d * rho^(d(d+1)/2).
//   - Structured (Thm thm:struct): every hop survives with constant fidelity q,
//     giving the steady per-hop cost P_ret^struct(d) = q^d.
//
// The simulator draws N seeded Bernoulli-chain trials, estimates P_ret(d) per
// arm empirically, and locates the empirical crossover g(d) = q^d - s0^d
// rho^(d(d+1)/2) > 0 (eq:gd, Prop prop:cross). It is the systems-flavored
// evidence that the closed-form theory formulas describe the generative model:
// with N large the empirical estimate matches the analytic value within a few
// binomial standard errors. Per-trial seeding (see trialSeed) makes every
// estimate independent of the goroutine schedule, so 1 worker and N workers
// return byte-identical counts.
//
// Provenance for the calibration and formulas: paper sec:depth ("Synthetic
// operating point" s0=0.70, rho=0.67, q=0.97), eq:simceiling, thm:struct,
// eq:gd, prop:cross. Golden values cross-checked in sim_test.go come from the
// canonical golden pack (theory section A.1/A.3), asserted verbatim.
//
// Stdlib only (math, math/rand, sync); gofmt-clean.
package depthsim

import "math"

// Arm identifies a retrieval mechanism in the per-hop survival model.
type Arm int

const (
	// Similarity retrieval: hop k survives with probability s0*rho^k
	// (geometric decay, Assumption ass:geometric).
	Similarity Arm = iota
	// Structured (typed-store) traversal: every hop survives with the
	// constant per-edge fidelity q (Thm thm:struct).
	Structured
)

// String renders an Arm for CLI tables and test failure messages.
func (a Arm) String() string {
	switch a {
	case Similarity:
		return "similarity"
	case Structured:
		return "structured"
	default:
		return "arm(?)"
	}
}

// Model holds the per-hop survival parameters. Domain (golden pack note 4):
// s0 in (0,1], rho in (0,1], q in (0,1].
type Model struct {
	S0  float64 // base similarity retention at hop 1
	Rho float64 // per-hop similarity decay factor
	Q   float64 // per-edge structured fidelity
}

// PaperCalibration is the recurring synthetic operating point from the paper
// (sec:depth "Synthetic operating point"): s0 = 0.70, rho = 0.67, q = 0.97.
var PaperCalibration = Model{S0: 0.70, Rho: 0.67, Q: 0.97}

// hopSurvival returns the survival probability of hop k (1-indexed) under arm a.
// Similarity: s0*rho^k; Structured: q (constant, independent of k).
func (m Model) hopSurvival(a Arm, k int) float64 {
	if a == Structured {
		return m.Q
	}
	return m.S0 * math.Pow(m.Rho, float64(k))
}

// AnalyticPRet returns the closed-form retrieval probability P_ret(d), the
// ground truth the Monte-Carlo estimate must reproduce.
//
//	Similarity (eq:simceiling, Thm thm:sim): s0^d * rho^(d(d+1)/2)
//	Structured (Thm thm:struct):             q^d
//
// The triangular exponent T = d(d+1)/2 is evaluated as a float64 division, so it
// matches the Python reference (membench.theory.recovery.similarity_recovery,
// where d*(d+1)/2 is true division) to within a libm ULP.
func (m Model) AnalyticPRet(a Arm, d int) float64 {
	if a == Structured {
		return math.Pow(m.Q, float64(d))
	}
	t := float64(d*(d+1)) / 2
	return math.Pow(m.S0, float64(d)) * math.Pow(m.Rho, t)
}

// LogAnalyticPRet returns log P_ret(d) in natural log-space (paper: log form of
// eq:simceiling is concave with slope -> -infinity). Log-space stays finite where
// the linear form underflows to 0 (golden pack A.1: similarity at s0=0.5, rho=0.3,
// d=60 underflows to 0.0 in float64 while the log form is finite).
func (m Model) LogAnalyticPRet(a Arm, d int) float64 {
	if a == Structured {
		return float64(d) * math.Log(m.Q)
	}
	t := float64(d*(d+1)) / 2
	return float64(d)*math.Log(m.S0) + t*math.Log(m.Rho)
}

// Gap returns the structured advantage g(d) = P_ret^struct(d) - P_ret^sim(d)
// (eq:gd). It is unimodal in d and, for q < 1, positive on a contiguous depth
// interval. Under PaperCalibration g(1) = 0.501, g(2) ~= 0.7935 (golden A.1).
func (m Model) Gap(d int) float64 {
	return m.AnalyticPRet(Structured, d) - m.AnalyticPRet(Similarity, d)
}

// AnalyticCrossover returns the smallest depth in 1..maxDepth with g(d) > 0,
// i.e. the closed-form crossover where structure first overtakes similarity
// (Prop prop:cross). Under PaperCalibration this is d* = 1 because structure
// already wins at depth 1 (g(1) = 0.501 > 0; golden A.3). Returns 0 if structure
// never leads within maxDepth.
func (m Model) AnalyticCrossover(maxDepth int) int {
	for d := 1; d <= maxDepth; d++ {
		if m.Gap(d) > 0 {
			return d
		}
	}
	return 0
}

// splitMix64 mixes a 64-bit state into a well-distributed output (Steele,
// Lea & Flood, "Fast Splittable Pseudorandom Number Generators", OOPSLA 2014).
// It turns a small structured key (base seed, arm, depth, trial index) into a
// high-entropy seed so adjacent trials do not produce correlated streams.
func splitMix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// trialSeed derives the RNG seed for a single Monte-Carlo trial. It is a pure
// function of (baseSeed, arm, depth, trial) and — crucially — does NOT depend on
// which goroutine runs the trial. That is what makes the whole estimate
// scheduling-independent: any partition of the trial indices across any number
// of workers reseeds each trial identically, so the pooled success count is the
// same for 1 worker or N workers.
func trialSeed(baseSeed uint64, a Arm, depth, trial int) int64 {
	h := splitMix64(baseSeed + 0x9e3779b97f4a7c15)
	h = splitMix64(h + uint64(a) + 0x100)
	h = splitMix64(h + uint64(depth) + 0x200)
	h = splitMix64(h + uint64(trial))
	return int64(h)
}

// Estimate is the pooled Monte-Carlo result for one (arm, depth): each trial is
// a Bernoulli draw with success probability P_ret(d), so Successes ~ Binomial(
// Trials, P_ret(d)).
type Estimate struct {
	Arm       Arm
	Depth     int
	Successes int
	Trials    int
}

// PHat is the empirical retrieval probability, the unbiased estimator of
// P_ret(d). Returns 0 when there are no trials.
func (e Estimate) PHat() float64 {
	if e.Trials == 0 {
		return 0
	}
	return float64(e.Successes) / float64(e.Trials)
}

// StdErr is the binomial standard error sqrt(p_hat*(1-p_hat)/n) of PHat. A test
// asserts the empirical estimate lands within a few of these of the analytic
// value; the SE shrinks as 1/sqrt(n), which is why N=200k pins P_ret(d) tightly.
func (e Estimate) StdErr() float64 {
	if e.Trials == 0 {
		return 0
	}
	p := e.PHat()
	return math.Sqrt(p * (1 - p) / float64(e.Trials))
}
