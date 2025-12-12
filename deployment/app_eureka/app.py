
import streamlit as st

st.set_page_config(page_title="Main App", page_icon="🤖")

# ✅ Logout button in sidebar
if st.session_state.get("authenticated"):
    if st.sidebar.button("🚪 Wyloguj"):
        st.session_state.clear()
        st.success("Wylogowano pomyślnie! Odśwież stronę, aby zalogować się ponownie.")
        st.stop()

st.title("🔐 Logowanie")
st.write("Witaj w aplikacji! Użyj poniższych linków, aby przejść do innych stron:")

st.page_link("pages/1_Chatbot.py", label="💬 Chatbot", icon="💬")
st.page_link("pages/2_Data.py", label="🧾 Dane (edytowalne)", icon="📄")
st.page_link("pages/3_Save.py", label="💾 Zapis & Cache", icon="💾")
st.page_link("pages/4_Settings.py", label="⚙️ Ustawienia", icon="⚙️")

# Login form
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

username = st.text_input("Nazwa użytkownika")
password = st.text_input("Hasło", type="password")

if st.button("Zaloguj"):
    if username and password:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.success(f"Zalogowano jako {username}")
    else:
        st.error("Podaj nazwę użytkownika i hasło.")