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
