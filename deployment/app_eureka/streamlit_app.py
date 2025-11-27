import streamlit as st
from streamlit_star_rating import st_star_rating
from ollama_instance import get_response
from session_state import initialise_session_state, disable_chat_input
import pandas as pd
# Initialise session state and run the app
st.title("Ollama Chatbot powered by Streamlit😀")

initialise_session_state()


# Sample DataFrame
@st.cache_data(ttl="0.5 days")
def load_data():
    data = pd.DataFrame({
        "wybierz": [False, False, False, False, False],
        "ocena_wiersz": [0, 0, 0, 0, 0],
        "id_informacji": [22434, 939993, 3443, 8488484, 993939],
        "typ_podatku": ["vat", "pit", "cit", "mit", "kit"],
        "info_wiersz": ["", "b", "c", "d", "e"],
    })
    return data


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


st.title("Wybór podatku")
options = st.multiselect(
    "Wybierz typ podatku?",
    ["vat", "pit", "cit"],
    max_selections=1,
    accept_new_options=False,
)
if options is not None:
    st.write("Wybrano podatek:", options)

    st.title("Ocena poszczególnych interpetacji")
    # Create a DataFrameEditor instance
    editor = DataFrameEditor(load_data())
    selection = dataframe_with_selections(load_data(), editor)

    st.title("Ocena calościowa")
    st.title("Gwiazdy")

    



    # App title
    st.write("Rate in 3 levels (⭐ to ⭐⭐⭐). If you pick more than 3, we'll cap it at 3.")

    st.write("Please rate your experience (1–3 stars):")

    # Render the 3-star rating widget
    rating = st_star_rating(
        label="Your rating:",
        maxValue=3,          # Exactly 3 stars
        defaultValue=2,      # Optional: preselect 2 stars
        key="three_star_rating"
    )

    # Display the result
    if rating is not None:
        st.success(f"Thank you! You rated: {rating} / 3 stars")

    # Optional: Save rating to session state or file
    if st.button("Save Rating"):
        st.session_state["rating"] = rating
        st.write("Rating saved in session state:", st.session_state["rating"])





    # val = st.feedback("stars", key="three_star_feedback")  # returns 0..4 or None
    # if val is not None:
    #     # Map 0..4 (displayed 1..5) into 1..3 by capping at 2
    #     three_star = min(val, 2) + 1
    #     st.success(f"You rated: {three_star} / 3 stars")
    # sentiment_mapping =  {1: "Ok", 2: "Nie ok"}
    # selected = st.slider("Rating", 1, 2)
    # st.markdown(f"You selected {sentiment_mapping[selected - 1]} star(s).")

    # selected = st.feedback("stars")
    # st.write("Selected:", selected)
    # st.write("Mapping keys:", list(sentiment_mapping))
    # if 0 <= selected < len(sentiment_mapping):
    #     st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")
    # else:
    #     st.error("Invalid selection.")



    st.title("Kciuk")
    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
    selected = st.feedback("thumbs")
    st.write("Selected:", selected)
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

st.title("Tutaj jest model😀")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

print(st.session_state.messages)
# Chat input
if prompt := st.chat_input("Say something", disabled=st.session_state.processing, on_submit=disable_chat_input):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.spinner("AI myślę ..."):
        full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        ai_response = get_response(full_prompt)

    
    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)

        
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    # Re-enable chat input
    st.session_state.processing = False
    
    # Rerun the app to update the chat history
    st.rerun()
    
    
# st.title("Podejscie 2😀")

# def init_wybierz_column(df: pd.DataFrame) -> pd.DataFrame:
#     """Initialize the dataframe with a selection column and a hash column"""
#     df_with_selections = df.copy()
#     df_with_selections.insert(0, "wybierz", False)

#     # Create a hash column by concatenating all the columns
#     df_with_selections["hash"] = df_with_selections.apply(lambda x: md5("|".join(tuple(str(x))).encode()).hexdigest(), axis = 1)

#     # Initialize the selection status in the session state 
#     # with the hash as the key and the selection status as the value
#     if "wybierz_status" not in st.session_state:
#         st.session_state.wybierz_status = df_with_selections[["wybierz", "hash"]].set_index("hash")["wybierz"].to_dict()
    
#     return df_with_selections

# def filter_df(df: pd.DataFrame) -> pd.DataFrame:
#     """Filter the dataframe based on the lifespan range selected by the user"""
   
#     # simple filter example 
#     life_span_range2 = st.sidebar.slider("wybierz Lifespan range2", 0, 3, (0, 3))
#     filtered_df = df[df["ocena_wiersz"].between(*life_span_range2)]
#     if "wybierz_status" in st.session_state:
#         filtered_df["wybierz"] = filtered_df["hash"].map(st.session_state["wybierz_status"])

#     return filtered_df


# def dataframe_with_selections2(df: pd.DataFrame, init_value: bool = False) -> pd.DataFrame:
#     """Display the dataframe with a selection column and allow the user to wybierz rows"""

#     st.write("This is a fragment now 👇")
#     # Get dataframe row-selections from user with st.data_editor
#     edited_df = st.data_editor(
#         df,
#         height = "stretch",
#         hide_index=True,
#         column_config={"wybierz": st.column_config.CheckboxColumn("Wybierz do oceny",required=True),
#                        "info_wiersz": st.column_config.TextColumn("Info-interpetacja",max_chars=300),
#                        "ocena_wiersz": st.column_config.NumberColumn(
#                         "Ocena-intrepretacja",
#                         help="0-brak oceny 1-negatywna 2-neutralna 3-pozytywna",
#                         default = 0, 
#                         min_value=0,
#                         max_value=3,
#                         step=1,
#                         format="%d ⭐",
#         )
#                        },
#         disabled=[col for col in df.columns if col not in ("wybierz","ocena_wiersz","info_wiersz")]#df.columns,
#     )
    
#     # Update the selection status in the session state
#     if "wybierz_status" in st.session_state:
#         with st.expander(f"selection log"):
#             for _, (wybierz, hash) in edited_df[['wybierz', "hash"]].iterrows():
#                 st.session_state["wybierz_status"][hash] = wybierz
                
#                 if wybierz:
#                     st.info(f"{hash} is wybierz")
        

#     # Filter the dataframe using the temporary column, then drop the column
#     selected_rows = edited_df[edited_df.wybierz]
#     return selected_rows.drop('wybierz', axis=1)

# df = init_wybierz_column(df)
# filtered_df = filter_df(df)
# selection = dataframe_with_selections2(filtered_df)
# st.button("⬇️ Save selection")
# st.write("Your selection:")
# st.write(selection)
