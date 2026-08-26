from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_INPUT_SHA256 = {
    "metadata": "96d68c52b5f36ef3ba1a3d0ee6ca879fce036fb449c5a689a2024377bf706e78",
    "QADD": "d24e799778bb10faa68667b365a62ae94b9d89c984a1fa06d760e9c1c0af136a",
    "QGAIN": "af829b293582ad5d50eb12eff7212a211b2ef4673b28011e18b8f24115dd947e",
    "QREV": "449e52d238cdcfcfa5920e7507c0fb8ae6699908019cbc10d6d2955e222acc88",
    "QCHAN": "6cbea32330994aba9bcfcf933e94d8e5844ded015d54d49b18bbde908f89259b",
    "QDIST": "e013d82cdfed9a2c696eaf4fd4eb31c148eada24c2c2ed1ff8a5ddf448175a39",
    "QTEMP": "76b428c5461b8ccb877fd99ac92a6f6600582be66466cc2c9e64cc181b772c97",
}

EXPECTED_STAGE2_ODDS_RATIOS = {
    "FAM_QADD_01": 2.771201,
    "FAM_QADD_02": 0.469584,
    "FAM_QADD_03": 0.321803,
    "FAM_QGAIN_02": 1.499591,
    "FAM_QGAIN_03": 2.065686,
    "FAM_QREV_01": 0.657813,
    "FAM_QCHAN_01": 2.200030,
}


class PublicationVerificationError(RuntimeError):
    pass


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _close(observed: float, expected: float, *, atol: float = 1e-6) -> bool:
    return math.isfinite(float(observed)) and math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=atol
    )


def _check(report: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    report.append({"check": name, "passed": bool(passed), "detail": detail})


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise PublicationVerificationError(f"Missing required output: {path}")
    return pd.read_csv(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PublicationVerificationError(f"Missing required output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_population(report: list[dict[str, Any]], table: pd.DataFrame, prefix: str) -> None:
    observed = {
        str(row.cohort): (int(row.recordings), int(row.participants))
        for row in table.itertuples(index=False)
    }
    expected = {"ALS": (418, 158), "CONTROL": (101, 66)}
    _check(report, f"{prefix}: population", observed == expected, f"observed={observed}")


def _verify_hashes(report: list[dict[str, Any]], table: pd.DataFrame, prefix: str) -> None:
    observed = {
        str(row.input): str(row.sha256).lower()
        for row in table.itertuples(index=False)
    }
    _check(
        report,
        f"{prefix}: frozen input hashes",
        observed == EXPECTED_INPUT_SHA256,
        f"observed={observed}",
    )


def verify_publication_outputs(outputs_root: str | Path) -> dict[str, Any]:
    outputs = Path(outputs_root).resolve()
    report: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Notebook 02: extent, persistence, cohort/longitudinal/bulbar results.
    # ------------------------------------------------------------------
    g1 = outputs / "extent_distribution_persistence"
    completion = _load_csv(g1 / "logs" / "completion_checklist.csv")
    _check(
        report,
        "N02: blocking completion",
        bool(_bool_series(completion["passed"]).all()) and len(completion) == 15,
        f"rows={len(completion)}",
    )
    _verify_population(
        report, _load_csv(g1 / "tables" / "population_by_cohort.csv"), "N02"
    )
    _verify_hashes(
        report, _load_csv(g1 / "manifests" / "input_inventory.csv"), "N02"
    )

    cohort = _load_csv(g1 / "tables" / "cohort_models.csv")
    non_exploratory_sig = cohort.loc[
        (pd.to_numeric(cohort["q_value"], errors="coerce") < 0.05)
        & ~cohort["inference_tier"].astype(str).eq("exploratory")
    ]
    _check(
        report,
        "N02: six non-exploratory FDR cohort findings",
        len(non_exploratory_sig) == 6,
        f"n={len(non_exploratory_sig)}",
    )

    longitudinal = _load_csv(g1 / "tables" / "als_longitudinal_models.csv")
    q = pd.to_numeric(longitudinal["q_value"], errors="coerce")
    average_change = longitudinal.loc[
        ~longitudinal["analysis"].astype(str).eq("longitudinal_positive_extent")
    ]
    average_q = pd.to_numeric(average_change["q_value"], errors="coerce")
    _check(
        report,
        "N02: no FDR average longitudinal change",
        int((average_q < 0.05).sum()) == 0,
        f"significant={int((average_q < 0.05).sum())}",
    )
    highband = longitudinal.loc[
        longitudinal["feature"].astype(str).eq("qchan_highband_ratio_deficit")
        & longitudinal["analysis"].astype(str).eq("longitudinal_positive_extent")
    ]
    highband_ok = (
        len(highband) == 1
        and float(highband.iloc[0]["estimate"]) < 0
        and float(highband.iloc[0]["q_value"]) < 0.05
    )
    _check(
        report,
        "N02: conditional high-band extent decreases longitudinally",
        highband_ok,
        (
            "estimate/q="
            + (
                f"{highband.iloc[0]['estimate']}/{highband.iloc[0]['q_value']}"
                if len(highband) == 1
                else "missing"
            )
        ),
    )

    bulbar = _load_csv(g1 / "tables" / "bulbar_models.csv")
    primary_bulbar = bulbar.loc[
        bulbar["analysis"].astype(str).isin({"bulbar_continuous", "bulbar_presence"})
    ].copy()
    primary_bulbar["q_value"] = pd.to_numeric(primary_bulbar["q_value"], errors="coerce")
    between_n = int(
        (
            primary_bulbar["target_term"].astype(str).eq("bulbar_between")
            & (primary_bulbar["q_value"] < 0.05)
        ).sum()
    )
    within_n = int(
        (
            primary_bulbar["target_term"].astype(str).eq("bulbar_within")
            & (primary_bulbar["q_value"] < 0.05)
        ).sum()
    )
    _check(
        report,
        "N02: bulbar FDR finding counts",
        (between_n, within_n) == (4, 2),
        f"between={between_n}; within={within_n}",
    )
    pause_within = primary_bulbar.loc[
        primary_bulbar["feature"].astype(str).eq("qadd_pause_ac_level_dbfs_median")
        & primary_bulbar["target_term"].astype(str).eq("bulbar_within")
    ]
    pause_ok = (
        len(pause_within) == 1
        and _close(float(pause_within.iloc[0]["estimate"]), 0.103156, atol=1e-5)
        and _close(float(pause_within.iloc[0]["q_value"]), 0.02864842, atol=1e-6)
    )
    _check(
        report,
        "N02: QADD within-bulbar sign/value",
        pause_ok,
        (
            f"estimate={pause_within.iloc[0]['estimate']}; q={pause_within.iloc[0]['q_value']}"
            if len(pause_within) == 1
            else "missing"
        ),
    )

    # ------------------------------------------------------------------
    # Notebook 03: multidimensional structure.
    # ------------------------------------------------------------------
    g2 = outputs / "multidimensional_structure"
    completion = _load_csv(g2 / "logs" / "completion_checks.csv")
    _check(
        report,
        "N03: blocking completion",
        bool(_bool_series(completion["passed"]).all()) and len(completion) == 11,
        f"rows={len(completion)}",
    )
    _verify_population(
        report, _load_csv(g2 / "tables" / "population_by_cohort.csv"), "N03"
    )
    _verify_hashes(
        report, _load_csv(g2 / "manifests" / "input_inventory.csv"), "N03"
    )

    contrast = _load_csv(g2 / "tables" / "family_contrast_summary.csv").set_index("level")
    contrast_ok = (
        set(contrast.index) == {"between", "within"}
        and _close(contrast.loc["between", "delta"], 0.091824, atol=1e-6)
        and _close(
            contrast.loc["between", "permutation_p_value"],
            17 / 10001,
            atol=5e-9,
        )
        and int(contrast.loc["between", "n_valid_permutations"]) == 10000
        and _close(contrast.loc["within", "delta"], 0.136333, atol=1e-6)
        and _close(
            contrast.loc["within", "permutation_p_value"],
            2 / 10001,
            atol=5e-9,
        )
        and int(contrast.loc["within", "n_valid_permutations"]) == 10000
    )
    _check(report, "N03: family contrasts", contrast_ok, contrast.to_json())

    pca = _load_csv(g2 / "tables" / "pca_components_and_retention.csv")
    pca["retained"] = _bool_series(pca["retained_stable_component"])
    retained = pca.groupby("level")["retained"].sum().astype(int).to_dict()
    pc1 = pca.loc[pca["component"].eq(1)].set_index("level")[
        "variance_explained_percent"
    ]
    pca_ok = (
        retained == {"between": 3, "within": 4}
        and _close(pc1.loc["between"], 21.971801, atol=1e-5)
        and _close(pc1.loc["within"], 20.871887, atol=1e-5)
    )
    _check(
        report,
        "N03: PCA retention and PC1 variance",
        pca_ok,
        f"retained={retained}; pc1={pc1.to_dict()}",
    )

    # ------------------------------------------------------------------
    # Notebook 04: production annotation audit.
    # ------------------------------------------------------------------
    human = outputs / "goal3_human_correspondence"
    s1 = _load_json(
        human / "stage1_annotation_audit" / "manifest" / "stage1_run_manifest.json"
    )
    s1_ok = (
        int(s1["csv_files"]) == 969
        and int(s1["unique_annotation_recordings"]) == 760
        and int(s1["unique_annotation_ratings"]) == 969
        and int(s1["multiple_rater_recordings"]) == 70
        and int(s1["raw_intervals"]) == 19184
        and int(s1["valid_intervals"]) == 19184
        and int(s1["parse_issues"]) == 0
        and int(s1["taxonomy_issues"]) == 0
        and int(s1["frozen_population_n"]) == 519
        and int(s1["annotation_recordings_in_frozen_population"]) == 516
        and s1["models_allowed"] is False
    )
    _check(report, "N04: annotation audit manifest", s1_ok, "frozen=519; annotated=516")

    # ------------------------------------------------------------------
    # Notebook 05: four-rater reliability.
    # ------------------------------------------------------------------
    s1b = _load_json(
        human
        / "stage1b_human_reference_reliability"
        / "manifest"
        / "stage1b_run_manifest.json"
    )
    s1b_ok = (
        bool(s1b["final_run"])
        and int(s1b["complete_four_rater_recordings"]) == 69
        and int(s1b["bootstrap_replicates"]) == 2000
        and _close(s1b["overall_family_observed_agreement"], 0.7818035426731079, atol=1e-12)
        and _close(s1b["overall_family_fleiss_kappa"], 0.43587229517222664, atol=1e-12)
        and _close(s1b["overall_family_gwet_ac1"], 0.6441759235032468, atol=1e-12)
        and _close(s1b["overall_item_observed_agreement"], 0.8903554175293306, atol=1e-12)
        and s1b["machine_comparison_enabled"] is False
    )
    _check(report, "N05: reliability manifest", s1b_ok, "n=69; bootstrap=2000")

    # ------------------------------------------------------------------
    # Notebook 06: construct-matched human correspondence.
    # ------------------------------------------------------------------
    s2root = human / "stage2_construct_matched_correspondence"
    s2 = _load_json(s2root / "manifest" / "stage2_run_manifest.json")
    s2_ok = (
        bool(s2["frozen_spec"]["final_run"])
        and int(s2["frozen_recordings"]) == 519
        and int(s2["frozen_participants"]) == 224
        and int(s2["production_human_recordings"]) == 516
        and int(s2["construct_pairs"]) == 21
        and int(s2["supported_construct_pairs"]) == 19
        and int(s2["primary_models_fit"]) == 19
        and int(s2["subject_collapsed_resamples"]) == 1000
        and sorted(s2["support_failed_pairs"]) == ["FAM_QREV_02", "FAM_QREV_03"]
        and _close(s2["specificity_exact_permutation_p"], 1 / 24, atol=1e-15)
    )
    _check(report, "N06: Stage-2 manifest", s2_ok, "19/21 supported; 1000 resamples")

    completion = _load_csv(s2root / "logs" / "stage2_blocking_completion_checklist.csv")
    _check(
        report,
        "N06: blocking completion",
        bool(_bool_series(completion["passed"]).all()) and len(completion) == 11,
        f"rows={len(completion)}",
    )

    primary = _load_csv(s2root / "tables" / "primary_construct_matched_models.csv")
    observed_or = (
        primary.loc[primary["pair_id"].isin(EXPECTED_STAGE2_ODDS_RATIOS)]
        .set_index("pair_id")["odds_ratio"]
        .to_dict()
    )
    or_ok = set(observed_or) == set(EXPECTED_STAGE2_ODDS_RATIOS) and all(
        _close(observed_or[key], expected, atol=1e-6)
        for key, expected in EXPECTED_STAGE2_ODDS_RATIOS.items()
    )
    _check(report, "N06: headline construct odds ratios", or_ok, f"observed={observed_or}")

    specificity = _load_csv(s2root / "tables" / "cross_family_specificity_summary.csv")
    specificity_ok = (
        len(specificity) == 1
        and int(specificity.iloc[0]["permutations"]) == 24
        and _close(
            specificity.iloc[0]["exact_family_label_permutation_p"],
            1 / 24,
            atol=1e-15,
        )
    )
    _check(report, "N06: exact family assignment test", specificity_ok, specificity.to_json())

    # ------------------------------------------------------------------
    # Notebook 07: item atlas / independent Stage-2 verification.
    # ------------------------------------------------------------------
    s3root = human / "stage3_item_mapping_atlas"
    s3 = _load_json(s3root / "manifest" / "mapping_atlas_run_manifest.json")
    s3_ok = (
        bool(s3["final_run"])
        and int(s3["frozen_recordings"]) == 519
        and int(s3["frozen_participants"]) == 224
        and int(s3["production_human_recordings"]) == 516
        and int(s3["human_items"]) == 28
        and int(s3["machine_indicators"]) == 19
        and int(s3["raw_correlation_rows"]) == 2660
        and int(s3["adjusted_within_family_pairs"]) == 74
        and int(s3["adjusted_within_family_models_fit"]) == 28
        and int(s3["stage2_regression_pairs_verified"]) == 19
    )
    _check(report, "N07: item-atlas manifest", s3_ok, "28 items; 19 indicators; 19 verified")

    completion = _load_csv(
        s3root / "logs" / "mapping_atlas_blocking_completion_checklist.csv"
    )
    _check(
        report,
        "N07: blocking completion",
        bool(_bool_series(completion["passed"]).all()) and len(completion) == 11,
        f"rows={len(completion)}",
    )

    verification = _load_csv(s3root / "logs" / "stage2_regression_verification.csv")
    max_est_diff = float(
        pd.to_numeric(verification["absolute_estimate_difference"], errors="coerce").max()
    )
    max_se_diff = float(
        pd.to_numeric(
            verification["absolute_standard_error_difference"], errors="coerce"
        ).max()
    )
    verify_ok = (
        len(verification) == 19
        and bool(_bool_series(verification["passed"]).all())
        and max_est_diff <= 1e-12
        and max_se_diff <= 1e-12
    )
    _check(
        report,
        "N07: independent Stage-2 regression verification",
        verify_ok,
        f"pairs={len(verification)}; max_est_diff={max_est_diff:.3g}; max_se_diff={max_se_diff:.3g}",
    )

    passed = all(row["passed"] for row in report)
    return {
        "analysis_outputs_root": str(outputs),
        "checks": report,
        "n_checks": len(report),
        "n_passed": sum(int(row["passed"]) for row in report),
        "status": "PASS" if passed else "FAIL",
    }
