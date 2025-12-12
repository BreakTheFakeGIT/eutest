from time import sleep
import streamlit as st
from modules.nav import nav
from modules.streamlit_style import HIDE_STYLE
from modules.streamlit_app_names import APP_page_title, APP_page_icon, APP_layout, APP_title

st.set_page_config(page_title=APP_page_title, page_icon=APP_page_icon, layout=APP_layout)
#st.markdown(HIDE_STYLE, unsafe_allow_html=True)

#st.set_page_config(layout='centered', page_icon='🏆', page_title='Vector-EUREKA')
#st.markdown(HIDE_STYLE, unsafe_allow_html=True)




def main():
    st.markdown(HIDE_STYLE, unsafe_allow_html=True)
    st.title(APP_title)
    st.write("Testowo, aby **ZALOGOWAĆ** się wpisz w polu **UŻYTKOWNIK** oraz **HASŁO** słowo:   ")

    username = st.text_input("**Użytkownik**")
    password = st.text_input("**Hasło**", type="password")

    if st.button("Zaloguj", type="primary"):
        if username == "" and password == "":
            st.session_state.logged_in = True
            st.success("Zalogowano pomyślnie!")
            sleep(0.5)
            nav()
            st.switch_page("pages/chat_eureka.py")
        else:
            st.error("Nieprawidłowy **użytkownik** lub **hasło**")

if __name__ == '__main__':
    main()
    