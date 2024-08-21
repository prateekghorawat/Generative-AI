import streamlit as st
from chatbot_utils import configure_retrieval_chain, MEMORY, DocumentLoader, create_agent, query_agent
from streamlit.external.langchain import StreamlitCallbackHandler

st.set_page_config(page_title="Unified Chatbot")
st.title("🫡 Chat with your Documents or CSV")

# Upload file (CSV or Document)
uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    type=list(DocumentLoader.supported_extensions.keys()),
    accept_multiple_files=True
)

# Chatbot Mode Selector
chat_mode = st.sidebar.radio(
    "Choose Chat Mode",
    ("Document Chatbot", "CSV Chatbot")
)

# For Document Chatbot
if chat_mode == "Document Chatbot":
    if not uploaded_files:
        st.info("Please upload documents to continue.")
        st.stop()

    use_compression = st.checkbox("compression", value=False)
    use_flare = st.checkbox("flare", value=False)
    use_moderation = st.checkbox("moderation", value=False)

    CONV_CHAIN = configure_retrieval_chain(
        uploaded_files,
        use_compression=use_compression,
        use_flare=use_flare,
        use_moderation=use_moderation
    )

    if st.sidebar.button("Clear message history"):
        MEMORY.chat_memory.clear()

    avatars = {"human": "user", "ai": "assistant"}

    if len(MEMORY.chat_memory.messages) == 0:
        st.chat_message("assistant").markdown("Ask me anything!")

    for msg in MEMORY.chat_memory.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    assistant = st.chat_message("assistant")
    if user_query := st.chat_input(placeholder="Ask your question..."):
        st.chat_message("user").write(user_query)
        container = st.empty()
        stream_handler = StreamlitCallbackHandler(container)
        with st.chat_message("assistant"):
            if use_flare:
                params = {"user_input": user_query}
            else:
                params = {"question": user_query, "chat_history": MEMORY.chat_memory.messages}
            response = CONV_CHAIN.run(params, callbacks=[stream_handler])
            if response:
                container.markdown(response)

# For CSV Chatbot
elif chat_mode == "CSV Chatbot":
    uploaded_file = uploaded_files[0] if uploaded_files else None
    if uploaded_file is not None:
        agent = create_agent(uploaded_file)
        query = st.text_area("Insert your query")
        if st.button("Submit Query"):
            response = query_agent(agent, query)
            st.write(response)
    else:
        st.info("Please upload a CSV file to continue.")
