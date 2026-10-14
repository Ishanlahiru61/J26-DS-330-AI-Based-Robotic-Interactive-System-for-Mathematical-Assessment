import unittest

from component2_app.questions import QuestionFactory


GAMES = ("COUNTING","COMPARISON","ORDERING","PLACE_VALUE","ADDITION","SUBTRACTION","MULTIPLICATION","DIVISION")


class QuestionFactoryTests(unittest.TestCase):
    def test_all_games_both_tiers_obey_hardware_and_answer_contract(self):
        ids = set()
        for game in GAMES:
            for tier in ("Tier A", "Tier B"):
                for seed in range(20):
                    q = QuestionFactory.generate(game, tier, seed)
                    self.assertLessEqual(len(q.numbered_cubes), 2)
                    self.assertTrue(all(0 <= value <= 9 for value in q.numbered_cubes))
                    self.assertIsInstance(q.expected_answer, int)
                    self.assertTrue(1 <= q.expected_answer <= 25)
                    self.assertNotIn("expected_answer", q.public_robot_task())
                    self.assertNotIn(q.question_id, ids); ids.add(q.question_id)

    def test_answers_are_mathematically_correct(self):
        for game in GAMES:
            for tier in ("Tier A", "Tier B"):
                q = QuestionFactory.generate(game, tier, 17)
                c = q.numbered_cubes
                if game == "COUNTING": expected = q.metadata["target_number"]
                elif game == "COMPARISON": expected = max(c)
                elif game == "ORDERING": expected = c[0] + q.metadata["sequence_step"]
                elif game == "PLACE_VALUE": expected = q.metadata["number"] // 10 if q.metadata["asked_place"] == "tens" else q.metadata["number"] % 10
                elif game == "ADDITION": expected = sum(c)
                elif game == "SUBTRACTION": expected = c[0] - c[1]
                elif game == "MULTIPLICATION": expected = c[0] * c[1]
                else: expected = c[0] // c[1]
                self.assertEqual(q.expected_answer, expected, (game, tier, q))

    def test_ordering_tiers_have_unambiguous_steps(self):
        a = QuestionFactory.generate("ORDERING", "Tier A", 4)
        b = QuestionFactory.generate("ORDERING", "Tier B", 4)
        self.assertEqual(a.numbered_cubes[1] - a.numbered_cubes[0], 2)
        self.assertEqual(b.numbered_cubes[1] - b.numbered_cubes[0], 4)
        self.assertEqual(b.metadata["sequence_step"], 2)
        self.assertIn("steps of 2", b.prompt)

    def test_counting_uses_one_numbered_reference_cube(self):
        q = QuestionFactory.generate("COUNTING", "Tier A", 8)
        self.assertEqual(q.numbered_cubes, (q.expected_answer,))
        self.assertIn(f"showing {q.expected_answer}", q.prompt)
        self.assertIn(f"Place {q.expected_answer} colored cubes", q.prompt)
        self.assertNotIn("colored_cube_count", q.metadata)

    def test_comparison_requires_colored_cube_representation(self):
        q = QuestionFactory.generate("COMPARISON", "Tier B", 8)
        self.assertEqual(q.expected_answer, max(q.numbered_cubes))
        self.assertIn("Which number is larger?", q.prompt)
        self.assertIn("Place that many colored cubes", q.prompt)
        self.assertNotIn("Choose the cube", q.prompt)

    def test_operation_prompts_name_operands_and_relationship(self):
        required_phrases = {
            "ADDITION": ("Add", "altogether"),
            "SUBTRACTION": ("take away", "left"),
            "MULTIPLICATION": ("equal groups", "in each group"),
            "DIVISION": ("Share", "equally between"),
        }
        for game, phrases in required_phrases.items():
            q = QuestionFactory.generate(game, "Tier B", 12)
            for value in q.numbered_cubes:
                self.assertIn(str(value), q.prompt)
            for phrase in phrases:
                self.assertIn(phrase, q.prompt)


if __name__ == "__main__": unittest.main()
