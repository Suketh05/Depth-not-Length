from scoring.tables import (
    ablation_table,
    depth_crossover_table,
    headline_table,
    model_robustness_table,
)


def _row(**kwargs):
    base = {
        "dataset": "dcbench",
        "arm": "none",
        "model": "claude-sonnet-4-6",
        "depth": 1,
        "spec_variant": "full",
        "compliance_rate": 0.0,
    }
    base.update(kwargs)
    return base


def test_headline_table_is_model_fixed_rows_are_arms():
    rows = [
        _row(arm="none", compliance_rate=0.2),
        _row(arm="brief", compliance_rate=0.9),
        _row(arm="brief", model="gpt-5.1", compliance_rate=0.99),  # excluded: not claude
        _row(arm="none", depth=2, spec_variant="stripped", compliance_rate=0.0),  # excluded: not native
    ]
    table = headline_table(rows)
    assert table[("dcbench", "none")] == 0.2
    assert table[("dcbench", "brief")] == 0.9
    assert ("dcbench", "none") in table and len(table) == 2


def test_headline_table_includes_longmemeval_regardless_of_depth():
    rows = [_row(dataset="longmemeval", arm="brief", depth=3, spec_variant="full", compliance_rate=0.5)]
    table = headline_table(rows)
    assert table[("longmemeval", "brief")] == 0.5


def test_model_robustness_table_is_brief_fixed_rows_are_models():
    rows = [
        _row(arm="brief", model="claude-sonnet-4-6", compliance_rate=0.9),
        _row(arm="brief", model="gpt-5.1", compliance_rate=0.7),
        _row(arm="none", model="claude-sonnet-4-6", compliance_rate=0.2),  # excluded: not brief
    ]
    table = model_robustness_table(rows)
    assert table[("dcbench", "claude-sonnet-4-6")] == 0.9
    assert table[("dcbench", "gpt-5.1")] == 0.7
    assert len(table) == 2


def test_depth_crossover_table_rows_are_arms_cols_are_depth():
    rows = [
        _row(arm="brief", depth=1, spec_variant="full", compliance_rate=0.9),
        _row(arm="brief", depth=2, spec_variant="stripped", compliance_rate=0.85),
        _row(arm="brief", depth=3, spec_variant="stripped", compliance_rate=0.8),
        _row(arm="mem0", depth=3, spec_variant="stripped", compliance_rate=0.1),
        _row(dataset="longmemeval", arm="brief", depth=1, compliance_rate=0.99),  # excluded
    ]
    table = depth_crossover_table(rows)
    assert table[("brief", 1)] == 0.9
    assert table[("brief", 3)] == 0.8
    assert table[("mem0", 3)] == 0.1
    assert ("brief", 1) in table and len(table) == 4


def test_ablation_table_relabels_the_four_cells():
    rows = [
        _row(arm="none", depth=1, spec_variant="full", compliance_rate=0.6),
        _row(arm="none", depth=3, spec_variant="stripped", compliance_rate=0.0),
        _row(arm="brief", depth=3, spec_variant="stripped", compliance_rate=0.6),
        _row(arm="random_context", depth=3, spec_variant="stripped", compliance_rate=0.0),
        _row(arm="mem0", depth=3, spec_variant="stripped", compliance_rate=0.3),  # excluded: not an ablation arm
    ]
    table = ablation_table(rows)
    assert table == {
        "full_spec/none": 0.6,
        "stripped/none": 0.0,
        "stripped/brief": 0.6,
        "stripped/random_context": 0.0,
    }
