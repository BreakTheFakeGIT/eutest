"""" home """
import streamlit as st
from modules.nav import nav
from modules.streamlit_style import HIDE_STYLE
from modules.streamlit_app_names import HOME_layout, HOME_page_icon, HOME_page_title
from modules.info import INFO, ZALOZENIA, FUNKCJE, PROBLEMY


st.set_page_config(page_title=HOME_page_title, page_icon=HOME_page_icon, layout=HOME_layout)
st.markdown(HIDE_STYLE, unsafe_allow_html=True)


def home():

    nav()

    st.header('🏡 O WektorChat-EUREKA')
    st.subheader('Informacje **OGÓLNE**', divider='rainbow')
    st.markdown(INFO)

    st.subheader('Podstawowe **ZAŁOŻENIA**', divider='rainbow')
    st.markdown(ZALOZENIA)

    st.subheader('Podstawowe **FUNKCJONALNOŚCI**', divider='rainbow')
    st.markdown(FUNKCJE)

    st.subheader('Możliwe **PROBLEMY TECHNICZNE**', divider='rainbow')
    st.markdown(PROBLEMY)

    st.title('_AI_ is :blue[cool] :sunglasses:')


if __name__ == '__main__':
    home()