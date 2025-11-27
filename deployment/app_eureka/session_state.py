import streamlit as st

def initialise_session_state():
    """
    Initialise session state variables from streamlit
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

def disable_chat_input():
    """
    Disable chat input from streamlit
    """
    st.session_state.processing = True


