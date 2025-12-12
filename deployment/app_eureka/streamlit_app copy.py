
import streamlit as st
import pandas as pd
from datetime import datetime

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


def fake_dataframe(n_rows: int = 10):
    """Return a fake DataFrame with Polish sample content."""
    data = {
        "ID": [i + 1 for i in range(n_rows)],
        "Imię": ["Anna", "Piotr", "Krzysztof", "Maria", "Jan", "Ewa", "Tomasz", "Karolina", "Paweł", "Magda"][:n_rows],
        "Nazwisko": ["Kowalska", "Nowak", "Saternus", "Wiśniewska", "Zieliński", "Wójcik", "Cwalina", "Kaczmarek", "Lewandowski", "Duda"][:n_rows],
        "Stan": ["Nowe", "W toku", "Zamknięte", "W toku", "Nowe", "Zamknięte", "W toku", "Nowe", "W toku", "Zamknięte"][:n_rows],
        "Komentarz": [
            "Przykładowy wpis", "Wymaga weryfikacji", "Zatwierdzone", "Do poprawy", "Czeka na decyzję",
            "Zakończone", "Do przeglądu", "Spotkanie umówione", "W trakcie analizy", "Do akceptacji"
        ][:n_rows],
    }
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
    st.title("💬 Chatbot (styl konwersacji)")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

    # Chat input
    prompt = st.chat_input("Wpisz wiadomość (po polsku)...")
    if prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Very simple rule-based assistant response (replace with your LLM call)
        # For multi-user: you can include username in context, or route via user ID
        response = (
            "Dziękuję za wiadomość! "
            "To jest przykładowa odpowiedź chatbota. "
            "Chcesz przekształcić wiadomość do tabeli lub zapisać wynik?"
        )
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Show last assistant message immediately
        st.chat_message("assistant").write(response)

    st.caption("Podmień logikę odpowiedzi na połączenie do Twojego modelu (np. OpenAI, Azure OpenAI).")

# --------------------------
# Page: Editable Data
# --------------------------
elif page == "🧾 Dane (edytowalne)":
    require_auth()
    st.title("🧾 Edytowalna tabela danych")

    # Init or regenerate fake DF
    cols = st.columns([1, 1, 2])
    with cols[0]:
        regen = st.button("Załaduj / Odśwież przykładowe dane")
    with cols[1]:
        rows = st.number_input("Liczba wierszy", min_value=5, max_value=50, value=10, step=1)
    with cols[2]:
        st.caption("Użyj poniższych kontrolek, aby wygenerować lub edytować dane.")

    if regen or st.session_state.df is None:
        st.session_state.df = fake_dataframe(n_rows=int(rows))

    # Editable DF (with column config)
    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ID": st.column_config.NumberColumn(disabled=True),
            "Imię": st.column_config.TextColumn(),
            "Nazwisko": st.column_config.TextColumn(),
            "Stan": st.column_config.SelectboxColumn(options=["Nowe", "W toku", "Zamknięte"]),
            "Komentarz": st.column_config.TextColumn(),
        },
        key="editor_df",
    )

    st.session_state.df = edited_df  # keep edits

    # Quick stats
    st.subheader("📊 Szybkie statystyki")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Liczba rekordów", len(edited_df))
    with colB:
        st.metric("Nowe", int((edited_df["Stan"] == "Nowe").sum()))
    with colC:
        st.metric("W toku", int((edited_df["Stan"] == "W toku").sum()))

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
