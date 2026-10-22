"""Step 4g (part 2) - plain-language child report for teachers and parents.

Turns the keys-only explanation (explain_formula.py) into sentences, using a language file in
config/report_templates/<language>.json (English now; Sinhala/Tamil can be added as new files).

Rules:
- strengths and on-track skills come first;
- at most two main reasons; practice suggestions are Component 2 activities for the child's grade;
- skills not assessed are listed for the teacher only, never in the parent text;
- "insufficient" children get no result, only a request for another session;
- the words "dyscalculia" / "disorder" never appear (screening, not diagnosis; register F4).

Names are not stored in this component; the app passes the child's name from student_profiles.
Usage: python child_report.py [language]   (prints example reports for synthetic children)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_formula import ARTIFACTS, MODULE  # noqa: E402

TEMPLATES = MODULE / "config" / "report_templates"


def load_template(language: str = "en") -> dict:
    return json.loads((TEMPLATES / f"{language}.json").read_text(encoding="utf-8"))


def join(items: list[str], t: dict) -> str:
    """'a, b and c'; with semicolons when an item itself contains 'and' ('counting; adding and subtracting')."""
    if len(items) <= 1:
        return "".join(items)
    sep = "; " if any(f" {t['list_and']} " in i for i in items) else ", "
    last = "; " if sep == "; " else " "
    return sep.join(items[:-1]) + f"{last}{t['list_and']} " + items[-1]


def skill_name(skill: str, grade: int, t: dict, parent: bool = False) -> str:
    names = t["skill_names_parent" if parent else "skill_names"]
    if skill == "arithmetic" and grade == 3:
        return names["arithmetic_grade3"]
    return names[skill]


def teacher_report(exp: dict, t: dict, name: str | None = None) -> str:
    tt, grade = t["teacher"], exp["grade"]
    lines = [tt["title"].format(name=name or exp["student_id"], grade=grade, band=tt["band"][exp["band"]])]
    if exp["band"] == "insufficient":
        return "\n".join(lines + [tt["insufficient"]])

    names = lambda skills: join([skill_name(s, grade, t) for s in skills], t)  # noqa: E731
    if exp["strengths"]:
        lines.append(tt["strengths"].format(skills=names(exp["strengths"])))
    if exp["on_track"]:
        lines.append(tt["on_track"].format(skills=names(exp["on_track"])))
    support = [s for s, v in exp["skills"].items() if v.get("level") in ("needs_practice", "needs_extra_help")]
    if support:
        items = [tt["support_item"].format(skill=skill_name(s, grade, t), level=t["levels"][exp["skills"][s]["level"]])
                 for s in support]
        lines.append(tt["support"].format(items=join(items, t)))
    if exp.get("profile"):
        lines.append(tt["profile"].format(profile=tt["profile_names"][exp["profile"]]))
    if exp["band"] in ("monitor", "moderate", "high") and exp["main_reasons"]:
        lines.append(tt["main_reasons"].format(skills=names(exp["main_reasons"])))
    activities = list(dict.fromkeys(a for s in support for a in exp["skills"][s]["practise"]))
    if activities:
        lines.append(tt["practise"].format(activities=join([t["activity_names"][a] for a in activities], t)))
    lines.append(tt["next_step"][exp["band"]])
    if exp["not_assessed"]:
        lines.append(tt["not_assessed"].format(skills=names(exp["not_assessed"])))
    lines.append(tt["data_note"].format(**exp["data"]))
    return "\n".join(lines)


def parent_report(exp: dict, t: dict, name: str = "Your child") -> str:
    pt, grade = t["parent"], exp["grade"]
    if exp["band"] == "insufficient":
        return pt["insufficient"].format(name=name)
    names = lambda skills: join([skill_name(s, grade, t, parent=True) for s in skills], t)  # noqa: E731
    support = [s for s, v in exp["skills"].items() if v.get("level") in ("needs_practice", "needs_extra_help")]
    lines = []
    if exp["band"] == "low" and not support:
        lines.append(pt["low"].format(name=name))
    else:
        if exp["strengths"]:
            lines.append(pt["strengths"].format(name=name, skills=names(exp["strengths"])))
        if exp["on_track"]:
            lines.append(pt["on_track"].format(name=name, skills=names(exp["on_track"])))
    if support and exp["band"] != "low":
        lines.append(pt["support"].format(name=name, skills=names(support)))
        lines.append(pt["help"].format(skills=names(support)))
        lines.append(pt["next_step"])
    lines.append(pt["note"])
    return "\n".join(lines)


def main() -> None:
    language = sys.argv[1] if len(sys.argv) > 1 else "en"
    t = load_template(language)
    explanations = json.loads((ARTIFACTS / "child_explanations_formula_v1.json").read_text(encoding="utf-8"))
    for band in ("high", "moderate", "monitor", "low", "insufficient"):
        exp = next((e for e in explanations if e["band"] == band), None)
        if exp:
            print(f"===== {band} =====\n-- teacher --\n{teacher_report(exp, t)}\n-- parent --\n{parent_report(exp, t)}\n")


if __name__ == "__main__":
    main()
