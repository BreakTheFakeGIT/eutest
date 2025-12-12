"""" data frames  function"""
from uuid import uuid4
import streamlit as st
import numpy as np
import pandas  as pd
import modules.utils_text as ut

pd.options.mode.copy_on_write = False
pd.options.mode.chained_assignment = None

@st.cache_data(ttl=36000)
def add_cols_to_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    """ Funkcja """
    data['cosinus_sim_kat'] = pd.cut(data['cosinus_sim'], 7, labels=False)
    return data

@st.cache_data(ttl=36000)
def concate_lst_data_frame(lst_data :list[pd.DataFrame]) -> pd.DataFrame:
    """ Funkcja """
    data = [df.dropna(axis=1, how='all') for df in lst_data]
    data = pd.concat(data, axis=0, ignore_index=True)

@st.cache_data(ttl=36000)
def prepare_data_to_display_data(data : pd.DataFrame) -> pd.DataFrame:
    """ Funkcja """
    data['cosinus_sim'] = data.groupby('id_informacji')['cosinus_sim'].transform('mean')
    #data = data.drop(['tabela'], axis=1)
    data = data.drop_duplicates('id_informacji')
    data['url'] = [ut.create_url(id_informacji) for id_informacji in data['id_informacji']]
    data['teza'] = [teza.replace(u';', u' ') for teza in data['teza']]
    data = data.sort_values(by='cosinus_sim', ascending=False)
    data['lp'] = np.arange(0, len(data))+1
    data = data[['id_informacji', 'url', 'data_wydania','sygnatura','teza','slowa_kluczowe','tpod','cosinus_sim']]
    # data['ocena'] = 3
    # data = data[['lp', 'cosinus_sim', 'id_informacji', 'url', 'data_wydania','sygnatura','teza','ocena','tpod']]
    return data

@st.cache_data(ttl=36000)
def msg_data_to_display_data(msg_data):
    """ Funkcja """
    msg_data = pd.DataFrame.from_dict(msg_data)
    return prepare_data_to_display_data(msg_data)


def display_data(data: pd.DataFrame, nrows: int):
    """ Funkcja """
    edited_df = st.dataframe(data=data.head(nrows),
                column_config={
                        "lp": st.column_config.NumberColumn(
                        "LP",
                        help="Liczba porzadkowa dopasowania od najlepiej do najgorzej.",
                        format="%d"),
                        "cosinus_sim": st.column_config.NumberColumn(
                        "MIARA",
                        help="Miara cosinusowa - okresla bliskosc pomiedzy reprezentacja wektorowa wyrazow.",
                        format="%.6f",),
                         "tpod": st.column_config.TextColumn(
                        "PODATEK",
                        max_chars=5),
                        "id_informacji": st.column_config.NumberColumn(
                        "ID INFORMACJI",
                        format="%d"),
                        "data_wydania": st.column_config.DateColumn(
                        "DATA WYDANIA",
                        format="YYYY-MM-DD"),
                        "sygnatura": st.column_config.TextColumn(
                        "SYGNATURA",
                        max_chars=100),
                        "teza": st.column_config.TextColumn(
                        "TEZA",
                        help="Przycieto do **60 znakow**",
                        max_chars=100),
                        "url": st.column_config.LinkColumn(
                        "LINKI DO EUREKI",
                        help="https://eureka.mf.gov.pl/informacje/",
                        display_text="otworz link"),
                        "slowa_kluczowe": st.column_config.TextColumn(
                        "SLOWA KLUCZOWE",
                        help="Slowa kluczowe z Eureki.",
                        max_chars=200),
                        # "ocena": st.column_config.NumberColumn(
                        # "OCENA",
                        # help='Skala "😀": 5, "🙂": 4, "😐": 3, "🙁": 25, "😞": 1?',
                        # min_value=1,
                        # max_value=5,
                        # step=1,
                        # format="%d ⭐"
                        # ),
                },
                hide_index=True
    )
    return edited_df


if __name__ == '__main__':
    pass