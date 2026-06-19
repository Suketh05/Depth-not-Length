# contributing

thanks for taking a look. this is a research benchmark, so the one thing we really
care about is that the numbers stay honest and reproducible. keep that in mind and
the rest is easy.

## getting set up

you need [uv](https://github.com/astral-sh/uv) and python 3.10 or newer (we develop
on 3.12).

```sh
uv sync --extra dev --extra viz
make check          # ruff, mypy, pytest
```

the native kernels are optional. build them with `make -C native/c` if you want the
parity tests to run against the c reference.

## the rules that actually matter

these are not style preferences, they are what keeps the benchmark defensible:

- never weaken or skip a test to make ci green. fix the code instead.
- keep the fairness lock. every arm sees the same model, the same budget and the same
  tools. memory is the only thing that changes between arms.
- measured numbers and vendor reported numbers never share a column. if you add a
  competitor you ran yourself, it goes in the measured tier with its manifest. if you
  cite a published number, it goes in the reported tier with the citation.
- if a result does not favour the structured arm, report it. don't bury it, don't
  round it up. the synthetic crossover is where the mechanism is meant to show, the
  natural datasets are reported as they land.

## adding an arm

implement `write` and `retrieve` from `membench.retrieval.base.MemorySystem`, register
it with `@register_arm`, and add a test. arms never pick their own budget, the caller
passes it into `retrieve`.

## adding a dataset

implement a loader that returns `Task` objects with their depth labels, and wire it
into `membench.harness.load_tasks`. depth labels that go into published numbers need
the dual annotation pass, single pass heuristic labels are fine for development only.

## before you open a pr

run `make check`, and run the native parity tests if you touched a kernel. keep prs
small and focused, it makes review a lot quicker. there is a checklist in the pr
template that mirrors the rules above.
