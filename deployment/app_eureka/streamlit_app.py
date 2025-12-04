import streamlit as st
from streamlit_star_rating import st_star_rating
import pandas as pd
from datetime import datetime
from ollama_instance import get_response

st.set_page_config(page_title="Chatbot + Data Editor", page_icon="🤖", layout="wide")

# --------------------------
# Utilities & Initialization
# --------------------------
def init_state():
    defaults = {
        "authenticated": False,
        "username": None,
        "messages": [],           # chat messages [{"role":"user"/"assistant", "content": "..."}]
        "df": None,               # main dataframe
        "saved_at": None,         # last save time
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

class DataFrameEditor:
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.edited_df = df.copy()
        self.init_value = False
    
    def update_dataframe(self, new_df: pd.DataFrame):
        self.edited_df = new_df
    
    def has_changes(self) -> bool:
        return not self.edited_df.equals(self.original_df)

    def reset_original(self):
        self.original_df = self.edited_df.copy()


# Define and apply a simple filter

class DataFrameEditor:
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.edited_df = df.copy()
        self.init_value = False
    
    def update_dataframe(self, new_df: pd.DataFrame):
        self.edited_df = new_df
    
    def has_changes(self) -> bool:
        return not self.edited_df.equals(self.original_df)

    def reset_original(self):
        self.original_df = self.edited_df.copy()


# Define and apply a simple filter
def dataframe_with_selections(df: pd.DataFrame, editor: DataFrameEditor) -> pd.DataFrame:
    df_with_selections = df.copy()
    # Add wybierz column to the DataFrame
    df_with_selections['wybierz'] = editor.init_value

    placeholder = st.empty()
    # Get DataFrame row selections from user with st.data_editor
    edited_df = placeholder.data_editor(
        df_with_selections,
        width= "stretch",
        hide_index=True,
        column_config={"wybierz": st.column_config.CheckboxColumn("Wybierz do oceny",required=True),
                    "ocena_wiersz": st.column_config.NumberColumn(
                        "Ocena-intrepretacja",
                        help="0-brak oceny 1-negatywna 2-neutralna 3-pozytywna",
                        default = 0, 
                        min_value=0,
                        max_value=3,
                        step=1,
                        format="%d ⭐"
                        ),
                    "info_wiersz": st.column_config.TextColumn("Info-interpetacja",max_chars=500, width="large",),
                    },
        disabled=[col for col in df.columns if col not in ("wybierz","ocena_wiersz","info_wiersz")]#df.columns,
    )
    
    # Check for changes and update the editor object
    if not edited_df.equals(editor.edited_df):
        editor.update_dataframe(edited_df)
        editor.reset_original()

    # Filter the DataFrame using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.wybierz]
    print(selected_rows)
    with st.expander(f"Wybrano wnioski"):
        for _, (wybierz, id_informacji, info_wiersz) in selected_rows[['wybierz', "id_informacji", "info_wiersz"]].iterrows():   
            if wybierz:
                st.info(f"Interpetacja {id_informacji} Opis: {info_wiersz}")
    #st.write(list(edited_df.wybierz))
    selection = selected_rows.drop('wybierz', axis=1)
    st.write("Wybrano do zapisu:")
    st.write(selection)
    return selection
# def fake_dataframe(n_rows: int = 10):
#     """Return a fake DataFrame with Polish sample content."""
#     data = {
#         "ID": [i + 1 for i in range(n_rows)],
#         "Imię": ["Anna", "Piotr", "Krzysztof", "Maria", "Jan", "Ewa", "Tomasz", "Karolina", "Paweł", "Magda"][:n_rows],
#         "Nazwisko": ["Kowalska", "Nowak", "Saternus", "Wiśniewska", "Zieliński", "Wójcik", "Cwalina", "Kaczmarek", "Lewandowski", "Duda"][:n_rows],
#         "Stan": ["Nowe", "W toku", "Zamknięte", "W toku", "Nowe", "Zamknięte", "W toku", "Nowe", "W toku", "Zamknięte"][:n_rows],
#         "Komentarz": [
#             "Przykładowy wpis", "Wymaga weryfikacji", "Zatwierdzone", "Do poprawy", "Czeka na decyzję",
#             "Zakończone", "Do przeglądu", "Spotkanie umówione", "W trakcie analizy", "Do akceptacji"
#         ][:n_rows],
#     }
#     return pd.DataFrame(data)

@st.cache_data(ttl="0.5 days")
def fake_dataframe(n_rows: int = 10):
    data ={
        "wybierz": [False, False, False, False, False],
        "ocena_wiersz": [0, 0, 0, 0, 0],
        "id_informacji": [22434, 939993, 3443, 8488484, 993939],
        "typ_podatku": ["vat", "pit", "cit", "mit", "kit"],
        "info_wiersz": ["", "b", "c", "d", "e"],
    }
    data = {k: v[:n_rows] for k, v in data.items()}
    return pd.DataFrame(data)


def require_auth():
    """Guard pages that require authentication."""
    if not st.session_state.authenticated:
        st.warning("Zaloguj się, aby korzystać z aplikacji.")
        st.stop()


# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("Nawigacja")
page = st.sidebar.radio(
    "Wybierz stronę",
    ["🔐 Logowanie", "💬 Chatbot", "🧾 Dane (edytowalne)", "💾 Zapis & Cache", "⚙️ Ustawienia"],
    index=0
)

# --------------------------
# Page: Login
# --------------------------
if page == "🔐 Logowanie":
    st.title("🔐 Logowanie")
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Nazwa użytkownika")
    with col2:
        password = st.text_input("Hasło", type="password")

    login_btn = st.button("Zaloguj")
    if login_btn:
        # Replace with real authentication (e.g., OAuth, JWT, DB lookup)
        if username and password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success(f"Zalogowano pomyślnie jako **{username}**.")
        else:
            st.error("Podaj nazwę użytkownika i hasło.")

    st.info("To jest proste logowanie demo. W produkcji użyj rzeczywistej autoryzacji.")

# --------------------------
# Page: Chatbot
# --------------------------
elif page == "💬 Chatbot":
    require_auth()
    st.header("💬 Chatbot (styl konwersacji)")
    with st.container(border=True,):
        prompt = st.empty()
        prompt = prompt.text_area('**PROMPT** - wklej pytanie lub opis sprawy:',
                                        '',
                                        key='text',
                                        placeholder = 'pytanie lub opis sprawy',
                                        height = 200
                                        )
        #prompt = st.session_state["tmp_prompt"]
        cols = st.columns([2, 1],vertical_alignment="bottom")
        with cols[1]:
            type_tax = st.multiselect(
                    "**Wybierz typ podatku**",
                    ["vat", "pit", "cit"],
                    max_selections=1,
                    accept_new_options=False)
        with cols[0]:
            if type_tax is not None:
                st.info(f"Wybrano podatek: {type_tax}")
            else:
                st.warning('**Wybierz typ podatku.**', icon="⚠️")
                st.stop()

        cols = st.columns([1, 2],vertical_alignment="bottom")
        with cols[0]:
            if st.button('Wyślij',type='primary'):
                if not (prompt):
                    st.warning('**Brak wpisu treści w prompt. Nie możesz po prostu nic wysłać.**', icon="⚠️")
                    st.stop()
                if not type_tax:
                    st.warning('**Wybierz typ podatku.**', icon="⚠️")
                    st.stop()
        with cols[1]:
            st.caption("Opis przycisków: Wybierz typ podatku oraz wyślij.")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

    # Chat input
    #prompt = st.chat_input("Wpisz wiadomość (po polsku)...")
    if prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("AI myślę..."):
            full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            response = get_response(full_prompt)

        # Very simple rule-based assistant response (replace with your LLM call)
        # For multi-user: you can include username in context, or route via user ID
        # response = (
        #     "Dziękuję za wiadomość! "
        #     "To jest przykładowa odpowiedź chatbota. "
        #     "Chcesz przekształcić wiadomość do tabeli lub zapisać wynik?"
        # )
        #resposne = fake_dataframe.to_dict()
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Show last assistant message immediately
        st.chat_message("assistant").write(response)
# --------------------------
# Page: Editable Data
# --------------------------
elif page == "🧾 Dane (edytowalne)":
    require_auth()
    st.header("🧾 Edytowalna tabela danych")

    cols = st.columns([1, 2])
    with cols[0]:
        rows = st.number_input("Liczba wierszy", min_value=5, max_value=50, value=10, step=1)
    with cols[1]:
        st.caption("Użyj poniższych kontrolek, aby wygenerować lub edytować dane.")

    st.subheader("Ocena poszczególnych interpetacji")
    # Create a DataFrameEditor instance
    editor = DataFrameEditor(fake_dataframe(n_rows=int(rows)))
    edited_df = dataframe_with_selections(fake_dataframe(n_rows=int(rows)), editor)
    st.session_state.df = edited_df  # keep edits

    with st.container(border=True,):
        st.subheader("Ocena całościowa interpetacji")
        text = st.empty()
        text_rate = text.text_area('Opisz ogólną ocenę',
                                     '',
                                     key='text',
                                     placeholder = 'ocena całościowa interpetacji',
                                     height = 200
                                     )
        rating = st_star_rating(
            label="Skala:",
            maxValue=3,          # Exactly 3 stars
            defaultValue=2,      # Optional: preselect 2 stars
            key="three_star_rating"
        )

        # Display the result
        if rating is not None:
            st.success(f"Ocena : {rating} / 3 gwiazdki")

        # Optional: Save rating to session state or file
        if st.button("Zapisz rating"):
            st.session_state["rating"] = rating
            st.write("Zapisz rating:", st.session_state["rating"])

        st.title("Kciuk")
        sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
        selected = st.feedback("thumbs")
        st.write("Wybrano:", selected)
        st.write("Mapping keys:", list(sentiment_mapping))
        if selected is not None:
            match selected: 
                case 0:
                    st.error(f"You selected {sentiment_mapping[selected]} star(s).")
                case 1:
                    st.success(f"You selected {sentiment_mapping[selected]} star(s).")
        else:
            st.warning("Nic nie wybrano.")




    if st.button("⬇️ Zapisz wybrane", type="primary"):
        st.write("Why hello there")
    else:
        st.write("Goodbye")

# --------------------------
# Page: Save & Cache
# --------------------------
elif page == "💾 Zapis & Cache":
    require_auth()
    st.title("💾 Zapis danych oraz czyszczenie cache")

    st.write("Możesz zapisać aktualną tabelę i wyczyścić cache oraz sesję (symulacja czyszczenia cookies).")

    file_name = st.text_input("Nazwa pliku CSV", value="dane_edycja.csv")
    col1, col2 = st.columns(2)
    with col1:
        save_btn = st.button("Zapisz DataFrame do CSV")
    with col2:
        clear_btn = st.button("Wyczyść cache + sesję")

    if save_btn:
        df_to_save = st.session_state.get("df", None)
        if df_to_save is None or df_to_save.empty:
            st.error("Brak danych do zapisania. Przejdź do strony 'Dane (edytowalne)' i wygeneruj/edytuj dane.")
        else:
            df_to_save.to_csv(file_name, index=False)
            st.session_state.saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"Zapisano do pliku: **{file_name}** (czas: {st.session_state.saved_at})")

    if clear_btn:
        # Clear Streamlit caches & session (server-side)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()
        st.info("Cache i sesja zostały wyczyszczone. Odśwież stronę, aby zalogować się ponownie.")

    st.caption("Uwaga: faktyczne cookies w przeglądarce nie mogą być czyszczone po stronie serwera.")

# --------------------------
# Page: Settings
# --------------------------
elif page == "⚙️ Ustawienia":
    require_auth()
    st.title("⚙️ Ustawienia")

    st.write("Konfiguracje aplikacji (demo).")
    theme = st.selectbox("Motyw", ["System", "Jasny", "Ciemny"], index=0)
    rows_default = st.slider("Domyślna liczba wierszy dla przykładowych danych", 5, 50, 10)

    apply_btn = st.button("Zastosuj")
    if apply_btn:
        # You can persist settings to session or file/db
        st.session_state["default_rows"] = rows_default
        st.success("Zastosowano ustawienia (demo).")

    st.divider()
    st.subheader("Informacje o sesji")
    st.json({
        "authenticated": st.session_state.get("authenticated"),
        "username": st.session_state.get("username"),
        "saved_at": st.session_state.get("saved_at"),
        "df_rows": len(st.session_state.df) if st.session_state.get("df") is not None else None,
        "messages_count": len(st.session_state.messages),
    })
elif page == "💾 Zapis & Cache":
    require_auth()
    st.title("💾 Zapis danych oraz czyszczenie cache")

    st.write("Możesz zapisać aktualną tabelę i wyczyścić cache oraz sesję (symulacja czyszczenia cookies).")

    file_name = st.text_input("Nazwa pliku CSV", value="dane_edycja.csv")
    col1, col2 = st.columns(2)
    with col1:
        save_btn = st.button("Zapisz DataFrame do CSV")
    with col2:
        clear_btn = st.button("Wyczyść cache + sesję")

    if save_btn:
        df_to_save = st.session_state.get("df", None)
        if df_to_save is None or df_to_save.empty:
            st.error("Brak danych do zapisania. Przejdź do strony 'Dane (edytowalne)' i wygeneruj/edytuj dane.")
        else:
            df_to_save.to_csv(file_name, index=False)
            st.session_state.saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"Zapisano do pliku: **{file_name}** (czas: {st.session_state.saved_at})")

            # ✅ Add download button
            with open(file_name, "rb") as f:
                st.download_button(
                    label="📥 Pobierz zapisany plik CSV",
                    data=f,
                    file_name=file_name,
                    mime="text/csv"
                )

    if clear_btn:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()
        st.info("Cache i sesja zostały wyczyszczone. Odśwież stronę, aby zalogować się ponownie.")

    st.caption("Uwaga: faktyczne cookies w przeglądarce nie mogą być czyszczone po stronie serwera.")
