import streamlit as st
from data_science_utlis import query_agent, create_agent

# Initialize session state if not already done
if 'data_file' not in st.session_state:
    st.session_state.data_file = None
if 'query' not in st.session_state:
    st.session_state.query = ""

st.title("🫡 Chat with your CSV")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV")

# Update session state with the uploaded file
if uploaded_file is not None:
    st.session_state.data_file = uploaded_file

# Option to delete the existing file
if st.session_state.data_file is not None:
    if st.button("Delete Uploaded File"):
        st.session_state.data_file = None
        st.session_state.query = ""  # Clear the query when deleting the file

# Text area for query input
st.session_state.query = st.text_area("Insert your query", value=st.session_state.query)

# Submit button
if st.button("Submit Query"):
    if st.session_state.data_file is None:
        st.error("Please upload a CSV file before submitting a query.")
    else:
        agent = create_agent(st.session_state.data_file)
        response = query_agent(agent=agent, query=st.session_state.query)
        st.write(response)
