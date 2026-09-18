import csv as csv_module

import pandas as pd
import pytest

from src.experiments.eval_grid import (
    CsvAppender, load_completed_cells, summarise, run, ROW_FIELDS,
)


# --------------------------------------------------------------------------
# resumability primitives - no anomalib/torch needed
# --------------------------------------------------------------------------

def test_load_completed_cells_empty_when_no_file(tmp_path):
    assert load_completed_cells(tmp_path / "nope.csv") == set()


def test_csv_appender_writes_header_once(tmp_path):
    p = tmp_path / "out.csv"
    w = CsvAppender(p, ROW_FIELDS)
    w.write_rows([{f: 0 for f in ROW_FIELDS}])
    w.close()
    w2 = CsvAppender(p, ROW_FIELDS)  # reopen, appending
    w2.write_rows([{f: 1 for f in ROW_FIELDS}])
    w2.close()

    with open(p, newline="") as f:
        lines = list(csv_module.reader(f))
    assert lines[0] == ROW_FIELDS
    assert len(lines) == 3  # header + 2 data rows
    assert lines.count(ROW_FIELDS) == 1


def test_load_completed_cells_reads_written_rows(tmp_path):
    p = tmp_path / "out.csv"
    w = CsvAppender(p, ROW_FIELDS)
    row = {f: 0 for f in ROW_FIELDS}
    row.update(restorer="identity", detector="padim", category="carpet",
              family="noise", severity=3)
    w.write_rows([row])
    w.close()

    cells = load_completed_cells(p)
    assert ("identity", "padim", "carpet", "noise", 3) in cells


def _fake_row(restorer, detector, category, family, severity, kind, image_id,
             label, sc, sd, sm):
    return dict(restorer=restorer, detector=detector, category=category,
               family=family, severity=severity, anomaly_kind=kind,
               image_id=image_id, label=label, score_clean=sc,
               score_degraded=sd, score_method=sm)


def test_summarise_gap_closed_full_recovery(tmp_path):
    """A restorer whose method-scores equal the clean scores should get
    gap_closed == 1.0 for that cell."""
    p = tmp_path / "cells.csv"
    rows = []
    # 4 normals (label 0) + 4 anomalies (label 1): clean scores separate the
    # classes perfectly, degraded scores destroy that separation, method
    # scores (for restorer "perfect") restore it exactly.
    for i in range(4):
        rows.append(_fake_row("perfect", "padim", "carpet", "noise", 3, "good",
                              i, 0, sc=0.1, sd=0.5, sm=0.1))
    for i in range(4, 8):
        rows.append(_fake_row("perfect", "padim", "carpet", "noise", 3, "blob",
                              i, 1, sc=0.9, sd=0.5, sm=0.9))
    pd.DataFrame(rows).to_csv(p, index=False)

    summary = summarise(p)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row.gap_closed == pytest.approx(1.0, abs=1e-6)
    assert row.auroc_clean == pytest.approx(1.0, abs=1e-6)
    assert row.auroc_degraded == pytest.approx(0.5, abs=1e-6)


def test_summarise_gap_closed_zero_when_method_matches_degraded(tmp_path):
    p = tmp_path / "cells.csv"
    rows = []
    for i in range(4):
        rows.append(_fake_row("noop", "padim", "carpet", "noise", 3, "good",
                              i, 0, sc=0.1, sd=0.5, sm=0.5))
    for i in range(4, 8):
        rows.append(_fake_row("noop", "padim", "carpet", "noise", 3, "blob",
                              i, 1, sc=0.9, sd=0.5, sm=0.5))
    pd.DataFrame(rows).to_csv(p, index=False)

    summary = summarise(p)
    row = summary.iloc[0]
    assert row.gap_closed == pytest.approx(0.0, abs=1e-6)


def test_summarise_separates_restorers_within_shared_clean_degraded_baseline(tmp_path):
    p = tmp_path / "cells.csv"
    rows = []
    for restorer, sm_good, sm_bad in [("good_restorer", 0.1, 0.9), ("bad_restorer", 0.5, 0.5)]:
        for i in range(4):
            rows.append(_fake_row(restorer, "padim", "carpet", "noise", 3, "good",
                                  i, 0, sc=0.1, sd=0.5, sm=sm_good))
        for i in range(4, 8):
            rows.append(_fake_row(restorer, "padim", "carpet", "noise", 3, "blob",
                                  i, 1, sc=0.9, sd=0.5, sm=sm_bad))
    pd.DataFrame(rows).to_csv(p, index=False)

    summary = summarise(p).set_index("restorer")
    assert summary.loc["good_restorer", "gap_closed"] == pytest.approx(1.0, abs=1e-6)
    assert summary.loc["bad_restorer", "gap_closed"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# end-to-end smoke - real DetectorHarness (PaDiM), synthetic data, tiny grid
# --------------------------------------------------------------------------

def test_run_end_to_end_smoke_and_resumes(tmp_path, monkeypatch):
    pytest.importorskip("anomalib")
    pytest.importorskip("lightning")
    import src.experiments.eval_grid as eg
    monkeypatch.setattr(eg, "results_dir", lambda: tmp_path)

    kwargs = dict(
        categories=["carpet"], detectors=["padim"],
        restorers=["identity", "gaussian"], families=["noise"], severities=[3],
        n_train=6, n_test=4, size=64, smoke=True, seed=0, tag="smoke_grid",
    )
    csv_path = eg.run(**kwargs)
    df = pd.read_csv(csv_path)
    assert not df.empty
    assert set(df.restorer) == {"identity", "gaussian"}
    n_rows_first = len(df)

    # re-run with identical args - every cell already complete, no new rows
    csv_path2 = eg.run(**kwargs)
    df2 = pd.read_csv(csv_path2)
    assert len(df2) == n_rows_first

    summary = eg.summarise(csv_path)
    assert not summary.empty
    assert "gap_closed" in summary.columns
