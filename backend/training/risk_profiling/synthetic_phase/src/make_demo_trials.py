"""Write a small DEMO cube_assessment_trials file to show the schema and test the code.

DEMO ONLY. Every number below that describes how children perform (success
probabilities, response times, demo rates) is a PLACEHOLDER with no
source. These rows must never be used to train, calibrate or evaluate anything.
Real rows come from the pilot (team_interface I6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
OUTPUT = MODULE / "data" / "demo_cube_assessment_trials.csv"
SEED = 7

# Activities offered per grade, following the NIE grade tables (nie_curriculum_map.md Section 0).
ACTIVITIES_BY_GRADE = {
    1: ["counting", "comparison", "place_value", "addition", "subtraction"],
    2: ["counting", "comparison", "ordering", "place_value", "addition", "subtraction"],
    3: ["counting", "comparison", "ordering", "place_value", "addition", "subtraction",
        "multiplication", "division"],
}
AREA = {
    "counting": "number_sense", "place_value": "number_sense",
    "comparison": "number_relationships", "ordering": "number_relationships",
    "addition": "arithmetic_calculation", "subtraction": "arithmetic_calculation",
    "multiplication": "multiplicative_reasoning", "division": "multiplicative_reasoning",
}
# Demo children: (id, grade, age_months, PLACEHOLDER success probability)
CHILDREN = [
    ("DEMO-001", 1, 76, 0.85), ("DEMO-002", 1, 80, 0.55),
    ("DEMO-003", 2, 88, 0.80), ("DEMO-004", 2, 91, 0.50),
    ("DEMO-005", 3, 100, 0.85), ("DEMO-006", 3, 103, 0.45),
]
QUESTIONS_PER_SESSION = 20  # within C2's 16-24 ([C2-msg])
SESSIONS = 2


def expected_answer(activity: str, tier: str, grade: int, rng: np.random.Generator) -> int:
    top = {1: 9, 2: 20, 3: 99}[grade]
    if activity == "counting":  # set size independent of tier (Grade 1 counts to 20, [NIE-G1] 2.2)
        return int(rng.integers(1, 4) if rng.random() < 0.3 else rng.integers(4, 21))
    if activity in ("multiplication", "division"):
        return int(rng.integers(2, 21))
    return int(rng.integers(1, top + 1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for student, grade, age, p_correct in CHILDREN:
        for session in range(1, SESSIONS + 1):
            for position in range(1, QUESTIONS_PER_SESSION + 1):
                activity = rng.choice(ACTIVITIES_BY_GRADE[grade])
                tier = "A" if (grade == 1 or rng.random() < 0.5) else "B"
                expected = expected_answer(activity, tier, grade, rng)
                correct = int(rng.random() < p_correct)
                given = expected if correct else max(0, expected + int(rng.choice([-2, -1, 1, 2])))
                rt = float(np.round(rng.lognormal(np.log(5 if correct else 8), 0.4), 2))
                demo = int((not correct) or rt >= 15)  # C1 rule ([P-C1] p. 29)
                representation = None
                if activity == "comparison":
                    representation = "non_symbolic" if grade == 1 else rng.choice(["non_symbolic", "symbolic"])
                rows.append({
                    "data_origin": "demo",
                    "student_id": student,
                    "session_id": f"{student}-S{session}",
                    "session_index": session,
                    "question_id": f"Q{position:02d}",
                    "interaction_position": position,
                    "age_months": age + (session - 1),
                    "grade": grade,
                    "c2_activity": activity,
                    "c2_area": AREA[activity],
                    "difficulty_tier": tier,
                    "representation": representation,
                    "expected_answer": expected,
                    "given_answer": given,
                    "accuracy": correct,
                    "error_value": abs(expected - given),
                    "response_time_s": min(rt, 15.0),
                    "attempts": 1 if correct else int(rng.integers(2, 4)),
                    # DEMO-006: C1 did not log demos, to show the fallback rule (B1c).
                    "robot_demo_triggered": None if student == "DEMO-006" else demo,
                    "valid_trial": int(rng.random() > 0.05),
                    "behavior_label": rng.choice(["finger_counting", "other_hand_movement", "uncertain"]),
                })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"{len(rows)} DEMO trial rows -> {OUTPUT}")


if __name__ == "__main__":
    main()
