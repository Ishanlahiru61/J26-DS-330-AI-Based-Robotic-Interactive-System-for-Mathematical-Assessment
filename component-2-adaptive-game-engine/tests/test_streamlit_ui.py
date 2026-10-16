import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class StreamlitSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        os.environ["COMPONENT2_DB_PATH"] = str(Path(self.tmp.name) / "ui.db")

    def tearDown(self):
        os.environ.pop("COMPONENT2_DB_PATH", None)
        self.tmp.cleanup()

    def test_researcher_dashboard_loads_without_exception(self):
        app = AppTest.from_file(APP_PATH).run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertEqual(app.header[0].value, "Researcher dashboard")
        self.assertEqual(len(app.dataframe), 3)

    def test_child_view_uses_started_backend_session(self):
        app = AppTest.from_file(APP_PATH).run(timeout=30)
        app.button[0].click().run(timeout=30)
        app.radio[0].set_value("Child").run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertEqual(app.header[0].value, "Cube Challenge")
        self.assertEqual(app.slider[0].label, "Choose how many colored cubes to place")
        self.assertIn("Submit answer", [button.label for button in app.button])

    def test_next_challenge_records_unanswered_before_replacing_task(self):
        app = AppTest.from_file(APP_PATH).run(timeout=30)
        next(button for button in app.button if button.label == "Start / resume").click().run(timeout=30)
        app.radio[0].set_value("Child").run(timeout=30)
        next(button for button in app.button if button.label == "Next challenge").click().run(timeout=30)
        self.assertFalse(list(app.exception))
        with sqlite3.connect(os.environ["COMPONENT2_DB_PATH"]) as db:
            statuses = db.execute("SELECT validation_status FROM interactions WHERE learner_id='DEMO_001'").fetchall()
            profile_total = db.execute("SELECT SUM(valid_interactions) FROM skill_profiles WHERE learner_id='DEMO_001'").fetchone()[0]
            cancelled = db.execute("SELECT COUNT(*) FROM question_instances WHERE status='CANCELLED'").fetchone()[0]
        self.assertEqual(statuses, [("UNANSWERED",)])
        self.assertEqual(profile_total, 0)
        self.assertEqual(cancelled, 1)

    def test_c4_export_download_persists_after_generation(self):
        app = AppTest.from_file(APP_PATH).run(timeout=30)
        next(button for button in app.button if button.label == "Export current C4 profile").click().run(timeout=30)
        self.assertFalse(list(app.exception))
        downloads = app.get("download_button")
        self.assertEqual(len(downloads), 1)
        self.assertEqual(downloads[0].label, "Download C4 JSON")
        payload = app.session_state["c4_export"]
        self.assertEqual(payload["learner_id"], "DEMO_001")
        self.assertEqual(payload["schema_version"], "C2_TO_C4_v2.0.0")

    def test_create_grade_one_and_five_complete_repeated_questions_and_export(self):
        app = AppTest.from_file(APP_PATH).run(timeout=30)
        for grade in (1,5):
            lid = f'UI_G{grade}'
            next(x for x in app.text_input if x.label == 'Learner ID').set_value(lid)
            next(x for x in app.text_input if x.label == 'Display name').set_value(lid)
            next(x for x in app.selectbox if x.label == 'Grade').set_value(grade)
            next(x for x in app.button if x.label == 'Create learner').click().run(timeout=30)
            next(x for x in app.selectbox if x.label == 'Learner').set_value(f'{lid} — {lid}').run(timeout=30)
            next(x for x in app.button if x.label == 'Start / resume').click().run(timeout=30)
            app.radio[0].set_value('Child').run(timeout=30)
            for _ in range(4):
                self.assertFalse(list(app.exception))
                with sqlite3.connect(os.environ['COMPONENT2_DB_PATH']) as db:
                    answer=db.execute("SELECT q.expected_answer FROM question_instances q JOIN sessions s USING(session_id) WHERE s.learner_id=? AND q.status='ACTIVE'",(lid,)).fetchone()[0]
                app.slider[0].set_value(answer)
                next(x for x in app.button if x.label == 'Submit answer').click().run(timeout=30)
            app.radio[0].set_value('Researcher').run(timeout=30)
            next(x for x in app.button if x.label == 'Export current C4 profile').click().run(timeout=30)
            p=app.session_state['c4_export']['profile']
            self.assertEqual(p['grade'],grade)
            self.assertEqual(sum(s['attempts'] for s in p['skills']),4)
            self.assertFalse(list(app.exception))

    def test_paused_session_cannot_submit_from_child_view(self):
        app=AppTest.from_file(APP_PATH).run(timeout=30)
        next(x for x in app.button if x.label=='Start / resume').click().run(timeout=30)
        next(x for x in app.button if x.label=='Pause').click().run(timeout=30)
        app.radio[0].set_value('Child').run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertNotIn('Submit answer',[x.label for x in app.button])

    def test_research_replay_failure_is_explicit(self):
        app=AppTest.from_file(APP_PATH).run(timeout=30)
        app.radio[0].set_value('Research replay').run(timeout=30)
        app.text_input[0].set_value(str(Path(self.tmp.name)/'missing')).run(timeout=30)
        next(x for x in app.button if x.label=='Run frozen research replay').click().run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertIn('unavailable',app.error[0].value)


if __name__ == "__main__": unittest.main()
