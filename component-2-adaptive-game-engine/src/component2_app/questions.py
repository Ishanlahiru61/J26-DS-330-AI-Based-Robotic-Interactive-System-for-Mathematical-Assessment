from __future__ import annotations

import random
import uuid
from dataclasses import asdict, dataclass
from recommendation_engine.curriculum import grade_support, curriculum_details


@dataclass(frozen=True)
class Question:
    question_id: str
    game_type: str
    tier: str
    prompt: str
    numbered_cubes: tuple[int, ...]
    expected_answer: int
    robot_action: str
    metadata: dict

    def public_robot_task(self) -> dict:
        value = asdict(self)
        value.pop("expected_answer")
        value["numbered_cubes"] = list(self.numbered_cubes)
        value["voice_instruction_text"] = self.prompt
        value["placement_zones"] = self.metadata.get("zones", ["left", "right"][:len(self.numbered_cubes)])
        value["hardware_profile_version"] = "ROBOT_TWO_NUMBER_CUBES_v1"
        return value


class QuestionFactory:
    """Creates robot-safe questions. Correct answers remain private to C1."""

    @staticmethod
    def generate(game_type: str, tier: str, seed: int | str | None = None, grade: int | None = None) -> Question:
        if tier not in {'Tier A', 'Tier B'}:
            raise ValueError('Unknown tier')
        if grade is not None and not grade_support(game_type, tier, grade):
            raise ValueError('QUESTION_CURRICULUM_NOT_SUPPORTED')
        rng = random.Random(seed)
        hard = tier == "Tier B"
        cubes: tuple[int, ...]
        meta: dict = {}
        if game_type == "COUNTING":
            answer = rng.randint(6, 9) if hard else rng.randint(1, 5)
            cubes = (answer,)
            prompt = f"Look at the numbered cube showing {answer}. Place {answer} colored cubes."
            action = f"Present one numbered cube showing {answer}"
            meta = {"target_number": answer}
        elif game_type == "COMPARISON":
            if hard:
                low = rng.randint(1, 8); values = [low, low + 1]
            else:
                a = rng.randint(1, 3); values = [a, rng.randint(a + 2, 5)]
            rng.shuffle(values); cubes = tuple(values); answer = max(values)
            prompt = f"The numbered cubes show {values[0]} and {values[1]}. Which number is larger? Place that many colored cubes."
            action = f"Present numbered cubes {values[0]} and {values[1]}"
        elif game_type == "ORDERING":
            step = 2 if hard else 1
            start = rng.randint(1, 5 if hard else 7)
            answer = start + step; cubes = (start, start + 2 * step)
            if hard:
                prompt = f"Count in steps of {step} from {start} to {start + 2 * step}. Which number comes between them? Place that many colored cubes."
            else:
                prompt = f"Count by ones from {start} to {start + 2}. Which number comes between them? Place that many colored cubes."
            action = f"Present sequence endpoints {start} and {start + 2 * step}"
            meta = {"sequence_step": step}
        elif game_type == "PLACE_VALUE":
            tens = rng.randint(1, 9 if hard else 5); ones = rng.randint(1, 9)
            ask = rng.choice(("tens", "ones")) if hard else "tens"
            answer = tens if ask == "tens" else ones; cubes = (tens, ones)
            prompt = f"The Tens and Ones cubes make {10 * tens + ones}. How many {ask} are shown? Place that many colored cubes."
            action = f"Place digit {tens} in Tens and digit {ones} in Ones"
            meta = {"number": 10 * tens + ones, "asked_place": ask, "zones": ["tens", "ones"]}
        elif game_type == "ADDITION":
            if hard:
                a = rng.randint(5, 9); b = rng.randint(max(2, 11 - a), 9)
            else:
                ceiling = 6 if grade == 1 else 10
                a = rng.randint(1, 5); b = rng.randint(1, min(5, ceiling - a))
            answer = a + b; cubes = (a, b)
            prompt = f"Add {a} and {b}. How many altogether? Place that many colored cubes."
            action = f"Present numbered addends {a} and {b}; speak the addition instruction"
        elif game_type == "SUBTRACTION":
            a = rng.randint(7, 9) if hard else rng.randint(2, 6); b = rng.randint(1, a - 1)
            answer = a - b; cubes = (a, b)
            prompt = f"Start with {a} and take away {b}. How many are left? Place that many colored cubes."
            action = f"Present start number {a} and take-away number {b}; speak the subtraction instruction"
        elif game_type == "MULTIPLICATION":
            if hard:
                while True:
                    a, b = rng.randint(2, 5), rng.randint(2, 5)
                    if a * b <= 25 and max(a, b) > 2: break
            else:
                a, b = 2, rng.randint(1, 5)
            answer = a * b; cubes = (a, b)
            prompt = f"There are {a} equal groups with {b} cubes in each group. How many cubes altogether? Place that many colored cubes."
            action = f"Present {a} groups and {b} cubes per group; speak the grouping instruction"
            meta = {"group_count": a, "cubes_per_group": b}
        elif game_type == "DIVISION":
            if hard:
                divisor = rng.choice((3, 4)); quotient = rng.randint(1, 9 // divisor)
            else:
                divisor = 2; quotient = rng.randint(1, 4)
            dividend = divisor * quotient; answer = quotient; cubes = (dividend, divisor)
            prompt = f"Share {dividend} cubes equally between {divisor} groups. How many cubes are in each group? Place that many colored cubes."
            action = f"Present total {dividend} and {divisor} groups; speak the equal-sharing instruction"
            meta = {"dividend": dividend, "group_count": divisor}
        else:
            raise ValueError(f"Unknown game_type: {game_type}")
        if len(cubes) > 2 or any(not 0 <= x <= 9 for x in cubes) or not 1 <= answer <= 25:
            raise AssertionError("Question violates the two-cube hardware contract")
        if grade is not None:
            meta['curriculum'] = curriculum_details(game_type, tier, grade)
        return Question(str(uuid.uuid4()), game_type, tier, prompt, cubes, answer, action, meta)
