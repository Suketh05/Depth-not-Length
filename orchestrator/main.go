// Command orchestrator shards a membench sweep across CPU cores (or hosts).
//
// Running 20+ memory systems over several datasets, depths, and seeds is
// embarrassingly parallel: each (dataset, seed) shard is an independent `membench
// run`. This orchestrator fans those shards out across a worker pool and waits for
// all of them, which is what makes a full sweep tractable in wall-clock time. It
// shells out to the Python CLI (the single source of truth for results), so it adds
// parallelism without duplicating any benchmark logic.
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

type shard struct {
	dataset string
	seed    int
}

type result struct {
	shard shard
	err   error
	dur   time.Duration
}

func main() {
	datasets := flag.String("datasets", "synthetic,dcbench", "comma-separated dataset names")
	seeds := flag.Int("seeds", 3, "number of seeds per dataset (0..seeds-1)")
	budget := flag.Int("budget", 150, "retrieval token budget")
	outDir := flag.String("out", "results", "output directory for per-shard JSONL")
	workers := flag.Int("workers", runtime.NumCPU(), "concurrent workers")
	cli := flag.String("cli", "membench", "membench CLI executable")
	flag.Parse()

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}

	var shards []shard
	for _, d := range strings.Split(*datasets, ",") {
		d = strings.TrimSpace(d)
		if d == "" {
			continue
		}
		for s := 0; s < *seeds; s++ {
			shards = append(shards, shard{dataset: d, seed: s})
		}
	}

	fmt.Printf("dispatching %d shards across %d workers\n", len(shards), *workers)
	results := make(chan result, len(shards))
	sem := make(chan struct{}, *workers)
	var wg sync.WaitGroup

	for _, sh := range shards {
		wg.Add(1)
		go func(sh shard) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			out := filepath.Join(*outDir, fmt.Sprintf("shard_%s_%d.jsonl", sh.dataset, sh.seed))
			start := time.Now()
			cmd := exec.Command(*cli, "run",
				"--dataset", sh.dataset,
				"--seed", fmt.Sprint(sh.seed),
				"--budget", fmt.Sprint(*budget),
				"--out", out)
			cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
			results <- result{shard: sh, err: cmd.Run(), dur: time.Since(start)}
		}(sh)
	}

	go func() { wg.Wait(); close(results) }()

	failed := 0
	for r := range results {
		status := "ok"
		if r.err != nil {
			status = "FAIL: " + r.err.Error()
			failed++
		}
		fmt.Printf("  %s seed=%d  %-6s (%s)\n", r.shard.dataset, r.shard.seed, status, r.dur.Round(time.Millisecond))
	}
	if failed > 0 {
		fmt.Fprintf(os.Stderr, "%d shard(s) failed\n", failed)
		os.Exit(1)
	}
	fmt.Println("all shards complete")
}
