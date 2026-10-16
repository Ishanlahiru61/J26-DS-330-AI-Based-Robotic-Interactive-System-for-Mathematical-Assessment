import os
import streamlit as st


def render():
    st.header('Frozen ASSISTments research replay')
    st.info('Observed-task probabilities select illustrative support actions. Local cube games use the curriculum/history policy; local transfer remains unvalidated.')
    workspace = st.text_input('External ASSISTments workspace', value=os.environ.get('C2_RESEARCH_WORKSPACE', ''))
    if st.button('Run frozen research replay', disabled=not workspace):
        try:
            from component2_app.research_replay import replay
            with st.spinner('Verifying frozen artifacts and scoring saved observations'):
                st.session_state.replay_result = replay(workspace)
        except Exception as exc:
            st.session_state.pop('replay_result', None)
            st.error(f'Research replay unavailable: {exc}')
    if st.session_state.get('replay_result'):
        st.json(st.session_state.replay_result)
