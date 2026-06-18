from configs.loader import load_run_cells


def test_no_duplicate_cells():
    cells = load_run_cells()
    keys = [(c.dataset, c.arm, c.model_name, c.spec_variant, c.depth) for c in cells]
    assert len(keys) == len(set(keys))


def test_budget_is_identical_across_arms_within_a_dataset():
    cells = load_run_cells()
    by_dataset = {}
    for c in cells:
        by_dataset.setdefault(c.dataset, set()).add(c.budget_tokens)
    for dataset, budgets in by_dataset.items():
        assert len(budgets) == 1, f"{dataset} has inconsistent budgets: {budgets}"


def test_code_search_tool_and_retry_policy_identical_everywhere():
    cells = load_run_cells()
    assert len({c.code_search_tool for c in cells}) == 1
    assert len({c.retry_policy for c in cells}) == 1


def test_memory_line_covers_all_five_arms_per_dataset_at_native_condition():
    cells = load_run_cells()
    arms_by_dataset = {}
    for c in cells:
        if c.model_name == "claude" and c.depth == 1 and c.spec_variant == "full":
            arms_by_dataset.setdefault(c.dataset, set()).add(c.arm)
    for dataset in ("dcbench", "swebench", "longmemeval"):
        assert arms_by_dataset[dataset] == {"none", "fullcontext", "mem0", "brief", "random_context"}


def test_model_line_covers_all_three_models_with_brief_fixed():
    cells = load_run_cells()
    models_by_dataset = {}
    for c in cells:
        if c.arm == "brief" and c.depth == 1 and c.spec_variant == "full":
            models_by_dataset.setdefault(c.dataset, set()).add(c.model_name)
    for dataset in ("dcbench", "swebench", "longmemeval"):
        assert models_by_dataset[dataset] == {"claude", "gpt", "open_weight"}


def test_unimplemented_models_are_flagged():
    cells = load_run_cells()
    flags = {c.model_name: c.model_implemented for c in cells}
    assert flags["claude"] is True
    assert flags["gpt"] is False
    assert flags["open_weight"] is False


def test_depth_crossover_covers_depths_1_2_3_for_dcbench_and_swebench_only():
    cells = load_run_cells()
    depths_by_dataset = {}
    for c in cells:
        if c.model_name == "claude":
            depths_by_dataset.setdefault(c.dataset, set()).add(c.depth)
    assert depths_by_dataset["dcbench"] == {1, 2, 3}
    assert depths_by_dataset["swebench"] == {1, 2, 3}
    assert depths_by_dataset["longmemeval"] == {1}  # inert placeholder, not config-dialed
