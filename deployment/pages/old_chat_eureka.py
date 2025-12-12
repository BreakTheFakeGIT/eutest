"""" chat eureka"""
from time import sleep
from uuid import uuid4
import streamlit as st
#from streamlit_tags import st_tags
from streamlit_feedback import streamlit_feedback
import numpy as np
import pandas as pd

from modules.nav import nav
import modules.database as db
import modules.utils_data_frame as udb
import modules.utils_text as ut
from modules.streamlit_style import HIDE_STYLE
from modules.streamlit_app_names import PARAM_edycja_app, CHAT_EUREKA_page_title, CHAT_EUREKA_page_icon, CHAT_EUREKA_layout
st.set_page_config(page_title=CHAT_EUREKA_page_title, page_icon=CHAT_EUREKA_page_icon, layout=CHAT_EUREKA_layout)
st.markdown(HIDE_STYLE, unsafe_allow_html=True)

pd.options.mode.copy_on_write = False
pd.options.mode.chained_assignment = None

conn = db.init_connection()
model_silver = db.set_model_type(st.secrets.models.silver)
slownik = db.pkl_load_slownik(st.secrets.models.slowa_kluczowe)


def main():

    nav()
    st.header('💬 WektorChat-EUREKA')

    if 'question_state' not in st.session_state:
        st.session_state.question_state = False

    if 'text' not in st.session_state:
        st.session_state.text = ""

    if "temp" not in st.session_state:
        st.session_state["temp"] = ""



    if 'fbk' not in st.session_state:
        st.session_state.fbk = str(uuid4())

    if "chat_history_" not in st.session_state:
        st.session_state.chat_history_ = []

    if "fbk_history" not in st.session_state:
        st.session_state.fbk_history = []

    if "option_tpod" not in st.session_state:
        st.session_state.option_tpod = "Inne"

    if "option_tpod_history" not in st.session_state:
        st.session_state.option_tpod_history = []

    if "multiselect_keywords" not in st.session_state:
        st.session_state.multiselect_keywords = []

    if "multiselect_keywords_history" not in st.session_state:
        st.session_state.multiselect_keywords_history = []

    if "feedback" not in st.session_state:
        st.session_state.feedback = None

    system_message = '''You are an AI assistant'''

    if 'history' not in st.session_state:
        st.session_state['history'] = [{'role': 'system', 'content': system_message}]
        st.session_state['counters'] = [0, 1]

    # clear chat history
    if st.button('Wyczyść historię'):
        st.session_state['history'] = [st.session_state['history'][0]]
        st.session_state['counters'] = [0, 1]
        st.session_state.chat_history_ = []
        st.session_state.fbk = str(uuid4())
        st.session_state.fbk_history = []
        st.session_state.option_tpod = "Inne"
        st.session_state.option_tpod_history = []
        st.session_state.feedback = None
        st.session_state.multiselect_keywords = []
        st.session_state.multiselect_keywords_history = []
        st.session_state.text = ""
        st.session_state.temp = ""
        st.rerun()

    st.markdown(f"**Numer sesji:** {st.session_state.fbk}")

    def clear_text():
        st.session_state["temp"] = st.session_state["text"]
        st.session_state["text"] = ""

    def collect_history(fbk,fbk_history,option_tpod_history,multiselect_keywords_history) -> str:
        opt_tpod_hist = '|'.join(option_tpod_history)
        fbk_hist = '|'.join(fbk_history)
        multiselect_keywords_history = '|'.join([','.join([str(c) for c in lst]) for lst in multiselect_keywords_history])
        session_hist = {'fbk': str(fbk),
                'fbk_history': str(fbk_hist), 
                'option_tpod_history': str(opt_tpod_hist),
                'multiselect_keywords_history': str(multiselect_keywords_history)}
        return str(session_hist)

    def display_answer():
        """ display chat """
        for msg in st.session_state['history'][1:]:
            if msg['role'] == 'user':
                with st.chat_message('user'):
                    for i in msg['content']:
                        if i['type'] == 'text':
                            st.write(i['text'])

            else:
                with st.chat_message('assistant'):

                    msg_content = ''.join(['  ' + char if char == '\n' else char for char in msg['content']])
                    #print(msg_content[-1])
                    if msg['type'] == 'data_frame':
                        st.markdown('Asystent: ' + msg_content)
                        data = udb.msg_data_to_display_data(msg['data'])
                        udb.display_data(data, nrows=15)

                    elif msg['type'] == 'empty_data_frame':
                        st.markdown('Asystent: ')
                        st.error(msg_content, icon="🚨")

                    elif msg['type'] in ['long_text','short_text']:
                        st.markdown('Asystent: ')
                        st.warning(msg_content, icon="⚠️")

                    else:
                        msg_content = "**wyjątek**"
                        st.info('Asystent: ' + msg_content, icon="ℹ️")


    def fbcb(response):
        """Update the history with feedback.
        The question and answer are already saved in history.
        Now we will add the feedback in that history entry.
        """
        if st.session_state.chat_history_:
            last_entry = st.session_state.chat_history_[-1]  # get the last entry
            last_entry.update({'feedback': response})  # update the last entry
            st.session_state.chat_history_[-1] = last_entry  # replace the last entry

            st.write(st.session_state.chat_history_[-1]["feedback"])
            st.toast("✔️ Otrzymana informacja zwrotna!")

        # Create a new feedback by changing the key of feedback component.
        st.session_state.fbk = str(uuid4())

    def create_answer(history, type, content, data):
        """Add question/answer to history."""
        if history is None:
            return
        if type is None:
            return
        if content is None:
            return
        if data is None:
            return

        st.session_state.chat_history_.append({ "content": "answer-history"})
        st.session_state['history'].append({'role': 'assistant', 'type': type, 'content': content, 'data': data})
        st.session_state['counters'] = [i+2 for i in st.session_state['counters']]

    display_answer()
    ########################################################################
    with st.container(border=True,):
        text = st.empty()
        text_input = text.text_area('**PROMPT** - wklej pytanie lub opis sprawy:',
                                     '',
                                     key='text',
                                     placeholder = 'pytanie lub opis sprawy',
                                     height = 200
                                     )
        text_input = st.session_state["temp"]
        option_tpod = st.radio(" Wybierz **PODATEK**:",["Inne","VAT", "PIT"], horizontal=True, key='option_tpod')
        keywords = st.multiselect(label = 'Wybierz **SŁOWA KLUCZOWE**:',
                                options=slownik,
                                max_selections=10,
                                key="multiselect_keywords",
                                placeholder="wybierz SŁOWO i naciśnij **ENTER**")

        option_tpod_db = f'podatek: {option_tpod}, slowa: {keywords}'
        st.write("Wybrano: podatek - ", f'**{option_tpod}**', ", słowa kluczowe - ", f'**{keywords}**')

    if text_input and keywords and option_tpod:
        st.session_state.question_state = True

    with st.container(border=True,):
        if st.button('Wyślij',on_click=clear_text):
            if not (text_input):
                st.warning('**Brak wpisu treści w prompt. Nie możesz po prostu nic wysłać.**', icon="⚠️")
                st.stop()
            if option_tpod == 'Inne':
                st.warning('**Wybierz typ podatku.**', icon="⚠️")
                st.stop()
            if not keywords:
                st.warning('**Brak wpisanego słowa kluczowego. Wpisz słowo lub słowa kluczowe.**', icon="⚠️")
                st.stop()

            msg = {'role': 'user', 'content': []}
            if text_input:
                msg['content'].append({'type': 'text', 'text': text_input})
                st.session_state.fbk_history.append(st.session_state.fbk)
                st.session_state.option_tpod_history.append(st.session_state.option_tpod)
                st.session_state.multiselect_keywords_history.append(st.session_state.multiselect_keywords)
                print('SESJA')
                print(st.secrets.models.silver)
                print(st.secrets.models.slowa_kluczowe)
                print(option_tpod_db)
                print(st.session_state.fbk_history)
                print(st.session_state.option_tpod_history)
                print(st.session_state.multiselect_keywords_history)

                session_hist = collect_history(fbk=st.session_state.fbk,
                                fbk_history=st.session_state.fbk_history,
                                option_tpod_history = st.session_state.option_tpod_history,
                                multiselect_keywords_history = st.session_state.multiselect_keywords_history)

                db.store_db_text_input(conn,schema=st.secrets.db_credentials.schema,
                                    table='apl_tekst_wpisu',
                                    nr_uuid_sesji=st.session_state.fbk_history[0],
                                    licznik_sesji=st.session_state['counters'][0],
                                    znacznik_czasu_wpisu=ut.set_datetime_local(),
                                    tekst_wpisu=text_input,
                                    edycja_app=PARAM_edycja_app,
                                    hist_nr_uuid_sesji= session_hist,
                                    wybor_tpod = option_tpod_db)

                content_type, content, ls_field, ls_text = ut.proces_text(text_input=text_input,
                                                                            text_session=st.session_state['history'][1:])
                content = content + f' \nPodatek: **{option_tpod}** \nSłowa kluczowe: **{keywords}**'
                if content_type =='data_frame':
                    with st.spinner('Proszę czekać...'):
                        data_models, exec_time = db.get_data_embedding(model=model_silver,
                                            ls_field=ls_field,
                                            ls_text=ls_text,
                                            option_tpod=option_tpod,
                                            keywords=keywords,
                                            #query_add_words_to_db=slownik,
                                            schema=st.secrets.db_credentials.schema,
                                            table_vector_emb=st.secrets.db_credentials.tab_emb,
                                            table_info=st.secrets.db_credentials.tab_info,
                                            table_dict=st.secrets.db_credentials.tab_dict,
                                            conn=conn)
                        print(data_models.shape[0])
                        if data_models.empty:
                            content_type = 'empty_data_frame'
                            content = '**Brak wyszukania - brak wektorów/fraz w bazie lub wybór podatku lub słów kluczowych (brak powiązania)**'
                            #st.stop()
                        else:
                            content = content + f' \nCzas wyszukania: **{exec_time} sek**'
                            st.success("Ok!")
                            data_models = udb.prepare_data_to_display_data(data_models)
                            session_hist = collect_history(fbk=st.session_state.fbk,
                                        fbk_history=st.session_state.fbk_history,
                                        option_tpod_history = st.session_state.option_tpod_history,
                                        multiselect_keywords_history = st.session_state.multiselect_keywords_history)
                            db.store_db_tab_output(conn,
                                        schema=st.secrets.db_credentials.schema,
                                        table='apl_tabela_model',
                                        data=data_models,
                                        znacznik=ut.set_datetime_local(),
                                        param_edycja_app=PARAM_edycja_app,
                                        nr_uuid_sesji=st.session_state.fbk_history[0],
                                        hist_nr_uuid_sesji=session_hist,
                                        wybor_tpod=content)

                else:
                    data_models=pd.DataFrame()

            data = data_models.to_dict('records')
            st.session_state['history'].append(msg)
            history = (
                st.session_state['history']
                if st.session_state['history'][0]['content']
                else st.session_state['history'][1:])

            if st.session_state.question_state:
                create_answer(history=history, type=content_type, content=content, data=data)
                st.rerun()

    with st.container(border=True,):
        st.write('Ocena **CAŁOŚCIOWA WYSZUKANYCH** interpetacji:')
        sleep(0.1)
        feedback = streamlit_feedback(
                    feedback_type="faces",
                    optional_text_label="Wpisz ocenę i kliknij SUBMIT",
                    align="flex-start",
                    key=st.session_state.fbk,
                    on_submit=fbcb)
        st.session_state.feedback = feedback

        # sentiment_mapping = ["one", "two", "three", "four", "five"]
        # selected = st.feedback("faces")
        # if selected is not None:
        #     st.markdown(f"You selected {sentiment_mapping[selected]} star(s).") 
        session_hist = collect_history(fbk=st.session_state.fbk,
                            fbk_history=st.session_state.fbk_history,
                            option_tpod_history = st.session_state.option_tpod_history,
                            multiselect_keywords_history = st.session_state.multiselect_keywords_history)
        if st.session_state.feedback:
            db.store_db_score_input(conn,
                                    schema=st.secrets.db_credentials.schema,
                                    table='apl_ocena_wpisu',
                                    nr_uuid_sesji=st.session_state.fbk_history[0],
                                    licznik_sesji=st.session_state['counters'][0],
                                    znacznik_ocena_wpisu=ut.set_datetime_local(),
                                    ocena_wpisu=feedback.get('score'),
                                    tekst_ocena_wpisu=feedback.get('text'),
                                    edycja_app=PARAM_edycja_app,
                                    hist_nr_uuid_sesji= session_hist,
                                    wybor_tpod = option_tpod_db)


if __name__ == '__main__':
    main()
