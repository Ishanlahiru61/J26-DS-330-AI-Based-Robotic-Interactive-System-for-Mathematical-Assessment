from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


# Raw and participant-level files stay out of Git (see the repository .gitignore).
REPO = Path(__file__).resolve().parents[5]
RAW = REPO / "data" / "raw" / "decarli2020"
PROCESSED = REPO / "data" / "private" / "risk_profiling"
SEED = 42
TARGET = "diagnostic_group"
SOURCE_FILES = {
    "profile": "Complete_Dataframe_def.xlsx",
    "enumeration": "Exact_enumeration_task.xlsx",
    "symbolic": "Symbolic_comparison_task.xlsx",
    "working_memory": "Visuo_spatial_WM_task.xlsx",
}


def read_group_column(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    group_column = "Group" if "Group" in frame.columns else "group"
    if "Sub" not in frame.columns or group_column not in frame.columns:
        raise ValueError(f"{path.name} must contain Sub and a group column.")
    pairs = frame[["Sub", group_column]].dropna(subset=["Sub"]).drop_duplicates()
    inconsistent = pairs.groupby("Sub")[group_column].nunique()
    if inconsistent.gt(1).any():
        raise ValueError(f"{path.name} contains conflicting group labels for a subject.")
    return frame.rename(columns={group_column: "diagnostic_group"})


def summarize_trials(frame: pd.DataFrame, accuracy: str, response_time: str, prefix: str) -> pd.DataFrame:
    return (
        frame.groupby("Sub", as_index=False)
        .agg(
            **{
                f"{prefix}_accuracy": (accuracy, "mean"),
                f"{prefix}_response_time": (response_time, "mean"),
                f"{prefix}_trials": (accuracy, "count"),
            }
        )
    )


def validate_source_links(sources: dict[str, pd.DataFrame]) -> list[int]:
    subject_sets = {
        name: set(frame["Sub"].dropna().tolist())
        for name, frame in sources.items()
    }
    profile_subjects = subject_sets["profile"]
    for name, subjects in subject_sets.items():
        if subjects != profile_subjects:
            missing = sorted(profile_subjects - subjects)
            extra = sorted(subjects - profile_subjects)
            raise ValueError(
                f"Subject mismatch in {name}: missing={missing}, extra={extra}."
            )

    profile_groups = (
        sources["profile"][["Sub", "diagnostic_group"]]
        .drop_duplicates()
        .set_index("Sub")["diagnostic_group"]
    )
    for name, frame in sources.items():
        groups = (
            frame[["Sub", "diagnostic_group"]]
            .drop_duplicates()
            .set_index("Sub")["diagnostic_group"]
        )
        if not groups.sort_index().equals(profile_groups.sort_index()):
            raise ValueError(f"Diagnostic-group labels disagree with profile data in {name}.")

    return sorted(profile_subjects)


def stratified_subject_assignments(profile: pd.DataFrame) -> pd.DataFrame:
    assignments: list[tuple[object, str]] = []
    for label, group in profile.groupby("diagnostic_group", sort=True):
        subjects = sorted(group["student_id"].unique().tolist())
        random.Random(f"{SEED}:{label}").shuffle(subjects)
        if len(subjects) < 5:
            raise ValueError(f"Not enough subjects in class {label!r} to make three splits.")

        train_count = round(len(subjects) * 0.60)
        validation_count = round(len(subjects) * 0.20)
        train_count = min(train_count, len(subjects) - 2)
        validation_count = min(validation_count, len(subjects) - train_count - 1)
        train = subjects[:train_count]
        validation = subjects[train_count : train_count + validation_count]
        test = subjects[train_count + validation_count :]
        assignments.extend((subject, "train") for subject in train)
        assignments.extend((subject, "validation") for subject in validation)
        assignments.extend((subject, "test") for subject in test)

    return pd.DataFrame(assignments, columns=["student_id", "split"]).sort_values(
        "student_id"
    )


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    profile = read_group_column(RAW / SOURCE_FILES["profile"])
    enumeration = read_group_column(RAW / SOURCE_FILES["enumeration"])
    symbolic = read_group_column(RAW / SOURCE_FILES["symbolic"])
    working_memory = read_group_column(RAW / SOURCE_FILES["working_memory"])
    sources = {
        "profile": profile,
        "enumeration": enumeration,
        "symbolic": symbolic,
        "working_memory": working_memory,
    }
    subjects = validate_source_links(sources)

    if profile["Sub"].duplicated().any():
        raise ValueError("Participant profile workbook must have exactly one row per subject.")

    enumeration_summary = summarize_trials(
        enumeration, "Accuracy", "RTs", "enumeration"
    )
    symbolic_summary = summarize_trials(symbolic, "Accuracy", "RTs", "symbolic_trials")
    working_memory_summary = summarize_trials(
        working_memory, "Accuracy_WM", "RTs_WM", "working_memory"
    )

    integrated = profile.rename(
        columns={"Sub": "student_id", "diagnostic_group": TARGET}
    )
    for summary in (enumeration_summary, symbolic_summary, working_memory_summary):
        integrated = integrated.merge(
            summary.rename(columns={"Sub": "student_id"}),
            on="student_id",
            how="left",
            validate="one_to_one",
        )

    expected_classes = {"DD", "contr"}
    observed_classes = set(integrated[TARGET].dropna().unique())
    if observed_classes != expected_classes:
        raise ValueError(f"Expected diagnostic classes {expected_classes}, found {observed_classes}.")
    if integrated[TARGET].isna().any():
        raise ValueError("Some participants have no diagnostic-group target.")

    assignments = stratified_subject_assignments(integrated)
    merged = integrated.merge(
        assignments, on="student_id", how="left", validate="one_to_one"
    )
    if merged["split"].isna().any():
        raise AssertionError("Some participants were not assigned to a split.")

    split_sets = {
        name: set(merged.loc[merged["split"] == name, "student_id"])
        for name in ("train", "validation", "test")
    }
    if (
        split_sets["train"] & split_sets["validation"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["validation"] & split_sets["test"]
    ):
        raise AssertionError("Participant IDs overlap between splits.")

    integrated.to_csv(PROCESSED / "component3_real_labeled_participant_dataset.csv", index=False)
    assignments.to_csv(PROCESSED / "component3_real_participant_split_assignments.csv", index=False)
    for name in ("train", "validation", "test"):
        merged.loc[merged["split"] == name].drop(columns="split").to_csv(
            PROCESSED / f"component3_real_{name}_prepared.csv", index=False
        )

    summary = {
        "target_column": TARGET,
        "target_meaning": "Source-provided diagnostic group: DD versus control (contr).",
        "target_is_mastery": False,
        "source_files": list(SOURCE_FILES.values()),
        "subjects": len(subjects),
        "source_linkage": "All source workbooks have identical subject IDs and consistent group labels.",
        "rows_per_split": {
            name: int((merged["split"] == name).sum())
            for name in ("train", "validation", "test")
        },
        "students_per_split": {
            name: len(ids) for name, ids in split_sets.items()
        },
        "target_counts_per_split": {
            name: merged.loc[merged["split"] == name, TARGET].value_counts().to_dict()
            for name in ("train", "validation", "test")
        },
        "warning": (
            "This is a small participant-level diagnostic-group dataset, not session-level "
            "mastery data. It can support a separate exploratory DD/control classification "
            "question but must not be presented as robotic-system mastery prediction."
        ),
        "models_trained": False,
    }
    (PROCESSED / "component3_real_target_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Integrated participant rows: {len(integrated)}")
    print(f"Target counts: {integrated[TARGET].value_counts().to_dict()}")
    print("Verified: source IDs and labels agree across all four workbooks.")
    print(f"Split counts: {summary['target_counts_per_split']}")
    print("Verified: participant-disjoint splits; no models trained.")
    print("Target is DD/control diagnostic group, not session mastery.")


if __name__ == "__main__":
    main()