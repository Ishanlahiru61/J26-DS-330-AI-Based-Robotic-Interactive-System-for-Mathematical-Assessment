from __future__ import annotations

import streamlit as st

from component2_app.mocks import MockC1, MockRobot


def render(service) -> None:
    st.header("Cube Challenge")
    session_id = st.session_state.get("active_session_id")
    learner_id = st.session_state.get("active_learner_id")
    if not session_id:
        st.info("Ask the researcher to start a session.")
        return
    if service.db.session(session_id)['status'] != 'ACTIVE':
        st.info('Session paused or ended. Resume it in the researcher view.')
        return
    if "task" not in st.session_state:
        st.session_state.task = service.next_task(session_id)
    task = st.session_state.task
    if task['robot_task'] is None:
        if task['recommendation']['recommendation_mode'] == 'ASSESSMENT_COMPLETE':
            st.success('The grade-applicable minimum assessment evidence is complete.')
        else:
            st.warning('No activity is currently available. Researcher review is required.')
        st.json(task['recommendation'])
        return
    st.caption(f"Grade {task['recommendation']['grade']} • {task['recommendation']['tier']} • {task['recommendation']['decision_type']} • {task['recommendation']['reason_code']}")
    public = MockRobot().present(task["robot_task"])
    st.subheader(public["prompt"])
    if public["game_type"] == "PLACE_VALUE":
        cols = st.columns(2)
        for col, label, value in zip(cols, ("Tens", "Ones"), public["numbered_cubes"]):
            col.metric(label, value)
    else:
        st.markdown(" ".join(f"### 🎲 {value}" for value in public["numbered_cubes"]))
    st.caption(public["robot_action"])
    answer = st.slider("Choose how many colored cubes to place", 0, 25, key=f"answer_{public['question_id']}")
    st.markdown("🟩 " * answer)
    if st.button("Submit answer", type="primary"):
        result = MockC1(service).validate_and_submit(learner_id, session_id, public["question_id"], answer)
        st.session_state.feedback = "Great work!" if result["validated_correctness"] else "Good try — let’s keep learning!"
        del st.session_state.task
        st.rerun()
    if st.session_state.get("feedback"):
        st.success(st.session_state.pop("feedback"))
    if st.button("Next challenge"):
        service.record_non_scoring(session_id, public["question_id"], "UNANSWERED")
        st.session_state.feedback = "Challenge skipped and recorded as unanswered."
        st.session_state.pop("task", None)
        st.rerun()
