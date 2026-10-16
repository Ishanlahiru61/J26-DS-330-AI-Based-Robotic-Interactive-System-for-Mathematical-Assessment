from __future__ import annotations

import streamlit as st


def render(service) -> None:
    st.header("Researcher dashboard")
    learners = service.db.learners()
    if not learners:
        st.warning("No synthetic learners found.")
        if st.button("Create demo learners"):
            service.seed_demo_learners(); st.rerun()
        return
    options = {f"{x['learner_id']} — {x['display_name']}":x["learner_id"] for x in learners}
    ids = list(options.values())
    active_id = st.session_state.get('active_learner_id')
    selected = st.selectbox("Learner", options, index=ids.index(active_id) if active_id in ids else 0, key="researcher_learner")
    learner_id = options[selected]
    if st.session_state.get("active_learner_id") != learner_id:
        st.session_state.pop("task", None); st.session_state.pop("active_session_id", None)
        st.session_state.pop("c4_export", None); st.session_state.pop("c4_export_learner", None)
        st.session_state.active_learner_id = learner_id
    a,b,c = st.columns(3)
    if a.button("Start / resume"):
        st.session_state.active_session_id = service.start_or_resume(learner_id)["session_id"]
    if b.button("Pause") and st.session_state.get("active_session_id"):
        service.pause(st.session_state.active_session_id)
    if c.button("End") and st.session_state.get("active_session_id"):
        service.end(st.session_state.active_session_id); st.session_state.pop("active_session_id", None); st.session_state.pop("task", None)
    with st.expander('Create a learner (Grades 1–5)'):
        with st.form('new_learner'):
            new_id = st.text_input('Learner ID')
            name = st.text_input('Display name')
            grade = st.selectbox('Grade', [1, 2, 3, 4, 5])
            if st.form_submit_button('Create learner'):
                try:
                    if any(x['learner_id'] == new_id for x in learners):
                        raise ValueError('That learner ID already exists; select it above.')
                    service.db.upsert_learner(new_id, name or new_id, grade)
                    st.success(f'Created {new_id}, Grade {grade}. Select the learner above.')
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    profile = service.profile(learner_id)
    st.caption(f"Grade {profile['grade']} • Descriptive performance; sample counts accompany every percentage")
    with st.expander('Curriculum eligibility and exclusions'):
        st.json(profile['coverage'])
    st.subheader("Eight game indices")
    st.dataframe([{k:v for k,v in row.items() if k in {"game_type","area","valid_interactions","correct_interactions","skill_index","confidence","last_assessed_at"}} for row in profile["skills"]], width="stretch")
    st.subheader("Four assessment areas")
    st.dataframe(profile["areas"], width="stretch")
    history = service.db.recent_game_history(learner_id)
    st.subheader("Recent validated interactions")
    st.dataframe(history[-10:], width="stretch")
    if st.session_state.get("task"):
        st.json(st.session_state.task["recommendation"], expanded=False)
    st.subheader("Research evidence")
    st.info("Frozen public-data model: H2 Gradient Boosting, RT-OFF, Platt calibrated. Test log loss 0.534442; Brier 0.178955; ROC-AUC 0.750158; ECE 0.003847. Local robot recommendations remain rule-based.")
    if st.button("Export current C4 profile"):
        st.session_state.c4_export = service.export_c4(learner_id, st.session_state.get("active_session_id"))
        st.session_state.c4_export_learner = learner_id
    if st.session_state.get("c4_export_learner") == learner_id and st.session_state.get("c4_export"):
        st.caption('Download is a snapshot at the export timestamp. Export again after further responses.')
        st.download_button("Download C4 JSON", data=__import__("json").dumps(st.session_state.c4_export, indent=2), file_name=f"{learner_id}_c4.json", mime="application/json")
        with st.expander('C4 descriptive JSON'):
            st.json(st.session_state.c4_export)
    confirm = st.checkbox("I understand reset affects synthetic learners only")
    if st.button("Reset synthetic demo data", disabled=not confirm):
        service.seed_demo_learners(); st.session_state.clear(); st.rerun()
