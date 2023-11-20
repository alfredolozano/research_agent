# main_streamlit_app.py

import streamlit as st
import callbacks
import chain_setup2  # Import the modified chain_setup module

QUESTION_HISTORY: str = 'question_history'

def init_stream_lit():
    title = "~0~0~0~0~0~"
    st.set_page_config(page_title=title, layout="wide")

    api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    if api_key:
        agent_executor = chain_setup2.setup_agent(api_key)
        st.header(title)
        if QUESTION_HISTORY not in st.session_state:
            st.session_state[QUESTION_HISTORY] = []

        simple_chat_tab, historical_tab = st.tabs(["Chat", "Session History"])
        with simple_chat_tab:
            user_question = st.text_input("- - -")
            with st.spinner('Please wait ...'):
                try:
                    response = agent_executor.run(user_question, callbacks=[callbacks.StreamlitCallbackHandler(st)])
                    st.write(f"{response}")
                    st.session_state[QUESTION_HISTORY].append((user_question, response))
                except Exception as e:
                    st.error(f"Error occurred: {e}")
        with historical_tab:
            for q in st.session_state[QUESTION_HISTORY]:
                question = q[0]
                if len(question) > 0:
                    st.write(f"Q: {question}")
                    st.write(f"A: {q[1]}")

if __name__ == "__main__":
    init_stream_lit()
