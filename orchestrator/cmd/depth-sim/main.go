// Command depth-sim prints the analytic-vs-empirical retrieval-probability table
// for the depth theory of "Depth, Not Length" (paper sec:depth).
//
// It runs the concurrent Monte-Carlo simulator in the depthsim package under the
// paper's synthetic operating point (s0=0.70, rho=0.67, q=0.97) and reports, per
// depth and per arm, the closed-form P_ret(d) (eq:simceiling / thm:struct), the
// empirical estimate, the deviation in binomial standard errors, and the
// structured-advantage gap g(d) (eq:gd). It closes with the analytic and
// empirical crossover depth (Prop prop:cross). This is a human-runnable witness
// that the theory formulas describe the generative model; paste its output into
// the PR body.
//
// Usage:
//
//	go run ./cmd/depth-sim [-n 200000] [-seed 7] [-workers 8] [-maxdepth 5]
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"

	"github.com/Suketh05/BriefBench/orchestrator/depthsim"
)

func main() {
	n := flag.Int("n", 200000, "Monte-Carlo trials per (arm, depth)")
	seed := flag.Uint64("seed", 7, "base seed (per-trial seeding makes results schedule-independent)")
	workers := flag.Int("workers", runtime.NumCPU(), "concurrent workers")
	maxDepth := flag.Int("maxdepth", 5, "maximum causal-hop depth")
	flag.Parse()

	m := depthsim.PaperCalibration
	fmt.Printf("depth-sim: paper calibration s0=%.2f rho=%.2f q=%.2f\n", m.S0, m.Rho, m.Q)
	fmt.Printf("N=%d trials/cell, seed=%d, workers=%d\n\n", *n, *seed, *workers)

	fmt.Printf("%-5s %-11s %-14s %-14s %-12s %-7s\n", "d", "arm", "analytic", "empirical", "|dev|", "sigma")
	fmt.Println("-------------------------------------------------------------------------")
	arms := []depthsim.Arm{depthsim.Similarity, depthsim.Structured}
	for d := 1; d <= *maxDepth; d++ {
		for _, a := range arms {
			est := m.SimulatePRet(a, d, *n, *workers, *seed)
			analytic := m.AnalyticPRet(a, d)
			se := math.Sqrt(analytic * (1 - analytic) / float64(*n))
			dev := math.Abs(est.PHat() - analytic)
			sigma := 0.0
			if se > 0 {
				sigma = dev / se
			}
			fmt.Printf("%-5d %-11s %-14.9g %-14.9g %-12.4g %-7.2f\n", d, a, analytic, est.PHat(), dev, sigma)
		}
		fmt.Printf("      g(%d) = q^%d - P_sim(%d) = %.9g\n", d, d, d, m.Gap(d))
	}

	fmt.Println()
	fmt.Printf("analytic crossover  (smallest d with g(d)>0): d* = %d\n", m.AnalyticCrossover(*maxDepth))
	fmt.Printf("empirical crossover (structured overtakes sim): d* = %d\n", m.EmpiricalCrossover(*maxDepth, *n, *workers, *seed))

	if *n <= 0 {
		fmt.Fprintln(os.Stderr, "note: n<=0 produces empty estimates")
	}
}
