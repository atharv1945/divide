import pandas as pd

from src.experiments.drr_study import (
    ROW_FIELDS, add_relative_drr, load_completed_cells, run, summarise, verdict,
)
from src.experiments.eval_grid import CsvAppender


# --------------------------------------------------------------------------
# resumability primitives
# --------------------------------------------------------------------------

def test_load_completed_cells_empty_when_no_file(tmp_path):
    assert load_completed_cells(tmp_path / "nope.csv") == set()


def test_load_completed_cells_reads_written_rows(tmp_path):
    p = tmp_path / "out.csv"
    w = CsvAppender(p, ROW_FIELDS)
    row = {f: 0 for f in ROW_FIELDS}
    row.update(category="carpet", image=2, family="noise", severity=3, restorer="identity")
    w.write_rows([row])
    w.close()

    cells = load_completed_cells(p)
    assert ("carpet", 2, "noise", 3, "identity") in cells


def test_add_relative_drr_on_empty_frame_is_a_noop():
    df = pd.DataFrame(columns=ROW_FIELDS)
    out = add_relative_drr(df)
    assert out.empty


def test_add_relative_drr_is_idempotent_on_a_csv_that_already_has_it():
    """Regression: a CSV read back after add_relative_drr already ran once
    (or written by the pre-resumability version of run(), which used to
    bake drr_identity/drr_rel into the CSV directly) must not make the
    pandas join collide on those columns the second time around."""
    row = {f: 0 for f in ROW_FIELDS}
    row.update(category="carpet", image=0, family="noise", severity=3,
               restorer="identity", drr=1.0)
    df = pd.DataFrame([row])
    once = add_relative_drr(df)
    twice = add_relative_drr(once)  # must not raise
    pd.testing.assert_frame_equal(once, twice)


# --------------------------------------------------------------------------
# end-to-end smoke - synthetic data, tiny grid, real resumability
# --------------------------------------------------------------------------

def test_run_end_to_end_smoke_and_resumes(tmp_path, monkeypatch):
    import src.experiments.drr_study as ds
    monkeypatch.setattr(ds, "results_dir", lambda: tmp_path)

    kwargs = dict(
        categories=["carpet"], restorers=["identity", "gaussian"],
        families=["noise"], severities=[3], kinds=["blob"],
        n_images=3, size=64, smoke=True, seed=0, tag="smoke_drr",
    )
    csv_path = run(**kwargs)
    df = pd.read_csv(csv_path)
    assert not df.empty
    assert set(df.restorer) == {"identity", "gaussian"}
    n_rows_first = len(df)

    # re-run with identical args - every cell already complete, no new rows
    csv_path2 = run(**kwargs)
    df2 = pd.read_csv(csv_path2)
    assert len(df2) == n_rows_first

    df = add_relative_drr(df2)
    assert "drr_rel" in df.columns
    summary = summarise(df)
    assert not summary.empty

    v, why = verdict(df)
    assert v in ("GO", "NO-GO", "MARGINAL", "INCONCLUSIVE")
    assert why


def test_run_only_computes_missing_cells_on_resume(tmp_path, monkeypatch):
    """The actual resumability contract: adding a restorer to an existing
    partial CSV must only compute the NEW restorer's cells, not redo the
    ones already on disk."""
    import src.experiments.drr_study as ds
    monkeypatch.setattr(ds, "results_dir", lambda: tmp_path)

    base_kwargs = dict(
        categories=["carpet"], families=["noise"], severities=[3], kinds=["blob"],
        n_images=3, size=64, smoke=True, seed=0, tag="smoke_drr_partial",
    )
    csv_path = run(restorers=["identity"], **base_kwargs)
    n_first = len(pd.read_csv(csv_path))

    csv_path2 = run(restorers=["identity", "gaussian"], **base_kwargs)
    df2 = pd.read_csv(csv_path2)
    assert len(df2) > n_first  # gaussian's cells were added
    assert (df2.restorer == "identity").sum() == n_first  # identity's rows untouched/not duplicated
