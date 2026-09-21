import numpy as np
import pandas as pd
import pytest

from src.experiments.drr_study import (
    ROW_FIELDS, _ratio_of_means, _ratio_of_means_by,
    add_relative_drr, load_completed_cells, run, summarise, verdict,
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
# ratio-of-means vs mean-of-ratios - the Step 0 fix
# --------------------------------------------------------------------------

def _rows(category_image_pairs, drr, drr_identity, restorer="wiener", family="defocus", severity=3):
    """Builds a minimal add_relative_drr-shaped frame: one identity row and
    one restorer row per (category, image), sharing drr_identity via the
    join key."""
    recs = []
    for (cat, img), d, d_id in zip(category_image_pairs, drr, drr_identity):
        recs.append(dict(category=cat, image=img, family=family, severity=severity,
                         restorer="identity", drr=d_id, anomaly_kind="scratch",
                         residual_corr=0.0, acg=1.0, dremr=0.0, psnr_normal=20.0))
        recs.append(dict(category=cat, image=img, family=family, severity=severity,
                         restorer=restorer, drr=d, anomaly_kind="scratch",
                         residual_corr=0.0, acg=1.0, dremr=0.0, psnr_normal=20.0))
    return add_relative_drr(pd.DataFrame(recs))


def test_ratio_of_means_differs_from_mean_of_ratios_on_a_small_denominator_example():
    """Regression for the exact discrepancy found in this project's own
    data: one example with a tiny drr_identity denominator blows up its
    individual ratio and drags mean-of-ratios far from what ratio-of-means
    (the now-chosen convention) reports on the same measurements. Three
    examples: two "normal" (drr=0.5, drr_identity=0.5, ratio 1.0 either
    way) and one with a near-zero identity denominator (drr=0.02,
    drr_identity=0.001, ratio 20.0) that a mean-of-ratios average would be
    dominated by."""
    pairs = [("c", 0), ("c", 1), ("c", 2)]
    drr = [0.5, 0.5, 0.02]
    drr_identity = [0.5, 0.5, 0.001]
    df = _rows(pairs, drr, drr_identity)

    non_id = df[df.restorer == "wiener"]
    ratio_of_means = _ratio_of_means(non_id)
    mean_of_ratios = float(np.mean([d / di for d, di in zip(drr, drr_identity)]))

    expected_rom = float(np.mean(drr) / np.mean(drr_identity))  # ~1.019 - two normal examples dominate
    assert ratio_of_means == pytest.approx(expected_rom, rel=1e-6)
    assert mean_of_ratios > ratio_of_means * 3  # the blow-up this fix removes (mean-of-ratios ~7.3)


def test_summarise_reports_ratio_of_means_not_mean_of_ratios():
    pairs = [("c", 0), ("c", 1), ("c", 2)]
    drr = [0.5, 0.5, 0.02]
    drr_identity = [0.5, 0.5, 0.001]
    df = _rows(pairs, drr, drr_identity)

    summary = summarise(df)
    expected = np.mean(drr) / np.mean(drr_identity)
    assert summary.loc["wiener", "drr_rel_mean"] == pytest.approx(expected, rel=1e-6)


def test_verdict_uses_ratio_of_means():
    pairs = [("c", i) for i in range(6)]
    drr = [0.1] * 5 + [0.02]
    drr_identity = [0.5] * 5 + [0.001]  # one near-zero-denominator example
    df = _rows(pairs, drr, drr_identity)

    v, why = verdict(df)
    expected = np.mean(drr) / np.mean(drr_identity)
    mean_of_ratios = float(np.mean([d / di for d, di in zip(drr, drr_identity)]))
    assert abs(expected - mean_of_ratios) > 1.0  # the two conventions really do diverge here
    assert f"{expected:.3f}" in why  # the reported number is ratio-of-means, not mean-of-ratios


def test_ratio_of_means_by_groups_correctly():
    """3 severity groups, not 2 - a real regression this project hit:
    make_figures()'s severity plot called _ratio_of_means_by(sub,
    ["severity"]) on real data with 3 severities and got a Series indexed
    by length-1 TUPLES on this pandas version (matplotlib then choked on
    plotting tuple x-values), while a 2-group fixture didn't reproduce it -
    keep this at 3+ groups so the regression stays covered."""
    pairs = [("c", 0), ("c", 1), ("c", 2), ("c", 3), ("c", 4), ("c", 5)]
    drr = [0.8, 0.8, 0.5, 0.5, 0.2, 0.2]
    drr_identity = [0.4] * 6
    df = _rows(pairs, drr, drr_identity, family="defocus")
    df.loc[df.restorer == "wiener", "severity"] = [2, 2, 3, 3, 4, 4]
    # re-join since severity changed after add_relative_drr already ran
    df = df.drop(columns=["drr_identity"])
    df.loc[df.restorer == "identity", "severity"] = [2, 2, 3, 3, 4, 4]
    df = add_relative_drr(df)

    non_id = df[df.restorer == "wiener"]
    by_sev = _ratio_of_means_by(non_id, ["severity"])
    assert list(by_sev.index) == [2, 3, 4]  # scalars, not (2,)/(3,)/(4,) tuples
    assert by_sev[2] == pytest.approx(0.8 / 0.4, rel=1e-6)
    assert by_sev[3] == pytest.approx(0.5 / 0.4, rel=1e-6)
    assert by_sev[4] == pytest.approx(0.2 / 0.4, rel=1e-6)
    import numpy as np
    np.asarray(by_sev.index, dtype=float)  # must not raise - what matplotlib does


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
    assert "drr_identity" in df.columns
    summary = summarise(df)
    assert not summary.empty
    assert "drr_rel_mean" in summary.columns

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
