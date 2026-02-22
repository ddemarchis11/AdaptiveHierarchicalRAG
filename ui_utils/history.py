import time
import datetime
import streamlit as st


def safe_sleep_for_rate_limit(min_time_between_requests: datetime.timedelta):
    now = datetime.datetime.now()
    if "prev_question_timestamp" not in st.session_state:
        st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

    if now - st.session_state.prev_question_timestamp < min_time_between_requests:
        time.sleep(0.2)

    st.session_state.prev_question_timestamp = now


def trim_history(history_length: int):
    msgs = st.session_state.get("messages", [])
    if len(msgs) > history_length * 2:
        st.session_state.messages = msgs[-history_length * 2 :]


def clear_conversation():
    st.session_state.messages = []
    st.session_state.initial_question = None
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)
