"""" navigation """
from time import sleep
from uuid import uuid4
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.source_util import get_pages
from modules.streamlit_style import HIDE_STYLE
from modules.streamlit_app_names import CHAT_EUREKA_page_title, CHAT_EUREKA_page_icon
from modules.streamlit_app_names import HOME_page_icon, HOME_page_title

def get_current_page_name():
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("Nie udało się uzyskać kontekstu skryptu")
    pages = get_pages("")
    return pages[ctx.page_script_hash]["page_name"]

def logout():
    st.session_state.logged_in = False
    st.info("Wylogowano pomyślnie!")
    sleep(0.5)
    st.switch_page("streamlit_app.py")


def nav():
    st.markdown(HIDE_STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.title("💎 Wyszukiwarka")
        st.write("")
        st.write("")

        if st.session_state.get("logged_in", False):

            st.sidebar.page_link('pages/home.py', label=HOME_page_title, icon=HOME_page_icon)
            st.sidebar.page_link('pages/chat_eureka.py', label=CHAT_EUREKA_page_title, icon=CHAT_EUREKA_page_icon)


            st.write("")
            st.write("")

            st.error('Pamiętaj o usunięciu \n\n**DANYCH OSOBOWYCH**', icon="🚨")

            st.write("")
            st.write("")

            if st.button("Wyloguj"):
                st.session_state['history'] = [st.session_state['history'][0]]
                st.session_state['counters'] = [0, 1]
                st.session_state.chat_history_ = []
                st.session_state.fbk_history = []
                st.session_state.feedback = None
                st.session_state.tpod_history = []
                st.session_state.option_tpod_history = []
                st.session_state.multiselect_keywords = []
                st.session_state.multiselect_keywords_history = []
                st.session_state.text = ""
                st.session_state.temp = ""
                logout()
                sleep(1)


        elif get_current_page_name() != "streamlit_app":
            # If anyone tries to access a secret page without being logged in,
            # redirect them to the login page
            st.switch_page("streamlit_app.py")


if __name__ == '__main__':
    nav()