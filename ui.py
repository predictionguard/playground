import json
import base64
import uuid
import hmac

import requests
import streamlit as st
from predictionguard import PredictionGuard
from langchain_community.document_loaders import PyPDFLoader
import lancedb
from langchain.text_splitter import CharacterTextSplitter
import pyarrow as pa
import pandas as pd


#-------------------#
# Authentication    #
#-------------------#

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop() 


#-------------------#
# PG setup          #
#-------------------#

# Create PredictionGuard client
client = PredictionGuard()

def stream_tokens(model, messages, system, temperature, max_new_tokens):
    for sse in client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system
            }
        ] + messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        stream=True
    ):
        yield sse["data"]["choices"][0]["delta"]["content"]


def gen_tokens(model, messages, system, temperature, max_new_tokens):
    result = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system
            }
        ] + messages,
        temperature=temperature,
        max_tokens=max_new_tokens
    )
    return result['choices'][0]['message']['content']


def embed(text):
    response = client.embeddings.create(
        model="multilingual-e5-large-instruct",
        input=text
    )
    return response['data'][0]['embedding']

def call_lvm(image, query, max_new_tokens):
    with open('/tmp/' + image.name, "wb") as f:
        f.write(image.read())
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": '/tmp/' + image.name,
                    }
                }
            ]
        },
    ]
    result = client.chat.completions.create(
        model="llava-1.5-7b-hf",
        messages=messages,
        max_tokens=max_new_tokens
    )
    result['choices'][0]['message']['content']

models = [
    "Hermes-3-Llama-3.1-8B",
    "Hermes-3-Llama-3.1-70B",
    "Neural-Chat-7B",
    "deepseek-coder-6.7b-instruct",
]


#--------------------------#
# Prompt templates, utils  #
#--------------------------#

default_system = "You are a helpful assistant that generally gives concise (1 to 2 sentence) responses unless asked to give longer or differently formatted responses."

qa_template = """Read the context below and respond with an answer to the question. If the question cannot be answered based on the context alone or the context does not explicitly say the answer to the question, write "Sorry I had trouble answering this question, based on the information I found."
 
Context: "{context}"
 
Question: "{query}"
"""

@st.cache_resource
def process_pdf(pdf_upload, chunk_size, chunk_overlap):
    
    # convert the pdf file in memory to path
    with st.spinner("Loading and parsing PDF..."):
        with open('/tmp/' +pdf_upload.name, "wb") as f:
            f.write(pdf_upload.read())
        loader = PyPDFLoader('/tmp/' + pdf_upload.name)
        pages = loader.load_and_split()
        text_out = ""
        for p in pages:
            text_out += p.page_content

    # process the document
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=' ')
    docs = text_splitter.split_text(text_out)

    # Get a unique id for the database
    uuid_raw = uuid.uuid4()

    # LanceDB setup
    uri = "/tmp/.lancedb-" + str(uuid_raw)
    db = lancedb.connect(uri)
    
    # Create a dataframe with the chunk ids, chunks, and embeddings
    with st.spinner("Embedding documents..."):
        data = []
        for i in range(len(docs)):
            if docs[i].strip() != "":
                data.append({
                    "chunk": i,
                    "text": docs[i],
                    "vector": embed(docs[i])
                })
    
    # Embed the documents
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 1024)),
        pa.field("text", pa.utf8()),
        pa.field("chunk", pa.int16()),
    ])
    table = db.create_table("docs", schema=schema)
    table.add(data)
    return table


#-----------------------#
# Streamlit UI          #
#-----------------------#

def home():
    st.title("Modaxo workshop, AI playground")
    st.image("pg_opea_components.jpg")
    st.markdown("""- `llm`: Generate text using Large Language Models (LLMs) like Hermes-2-Pro-Llama-3-8B, Hermes-2-Pro-Mistral-7B, Neural-Chat-7B, deepseek-coder-6.7b-instruct, etc.)
- `lvm`: Reason over images using Vision Models (LVMs) like LLaVA
- `embedding`: Generate embeddings for us in RAG workflows or semantic search applications
- `chat`: Chat with LLMs
- `rag q&a`: Answer questions with an LLM augmented by external data""")
    
# - Guardrails:
#     - `PII`: Detect and sanitize PII in text inputs
#     - `prompt injection`: Detect prompt injection attacks
#     - `factual consistency`: Score the factual consistency between a text (e.g., an LLM output) and reference data (e.g., in a RAG workflow)
#     - `toxicity`: Score the level of toxicity in a text (e.g., an LLM output)

def llm():
    st.title("Large Language Model (LLM)")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.1)
    max_new_tokens = st.slider("Max new tokens", 1, 2000, 1000)
    model = st.selectbox("Model", models)
    query = st.text_area("User message", height=100)
    if query:
        with st.spinner("Calling LLM..."):
            st.write_stream(stream_tokens(
                model, [{"role": "user", "content": query}], 
                default_system, temperature, max_new_tokens
            ))        

def lvm():
    st.title("Vision Model (LVM)")
    max_new_tokens = st.slider("Max new tokens", 1, 2000, 200)
    image = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    query = st.text_area("Query", height=100)
    if st.button("Submit/ Generate"):
        with st.spinner("Calling LVM..."):
            answer = call_lvm(image, query, max_new_tokens)
            st.write(answer)

def embedding():
    st.title("Embedding")
    text2embed = st.text_area("Text to embed", height=200)
    if text2embed:
        with st.spinner("Embedding..."):
            st.write(embed(text2embed))

def guardrails():
    st.title("Guardrails (aka safeguards)")

    with st.expander("PII"):
        piitext = st.text_area("Text", height=200)
        replace = st.checkbox("Replace PII")
        replace_method = st.selectbox("Replace method", ["random", "fake", "category", "mask"])
        if st.button("Identify and sanitize PII"):
            with st.spinner("Checking for PII..."):
                st.write("PII output:", call_pii(piitext, replace, replace_method))

    with st.expander("Prompt Injection"):
        pitext = st.text_area("Prompt", height=200)
        if st.button("Check for injection"):
            with st.spinner("Checking for injection..."):
                st.write("Injection probability:", call_pi(pitext))
    
    with st.expander("Factual consistency"):
        text1 = st.text_area("Text to check (e.g., output of an LLM)", height=200)
        text2 = st.text_area("Reference text", height=200)
        if st.button("Check"):
            with st.spinner("Checking factual consistency..."):
                st.write("Factuality score:", call_factuality(text1, text2))

    with st.expander("Toxicity"):
        toxtext = st.text_area("Text to inspect (e.g., output of an LLM)", height=200)
        if st.button("Check for toxicity"):
            with st.spinner("Checking toxicity..."):
                st.write("Toxicity score:", call_toxicity(toxtext))

def chat():
    st.title("LLM chatbot")
    with st.expander("Settings"):
        temperature = st.slider("Temperature", 0.0, 2.0, 0.1)
        max_new_tokens = st.slider("Max new tokens", 1, 2000, 1000)
        model = st.selectbox("Model", models)
        st.session_state["pg_model"] = model
        system = st.text_area("System prompt", height=100, value=default_system)
    if st.button("Reset chat"):
        st.session_state.messages = []
    st.divider()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):

            # generate response
            st.session_state['full_response'] = ""
            if len(st.session_state.messages) > 6:
                messages_to_use = st.session_state.messages[-5:].copy()
            else:
                messages_to_use = st.session_state.messages.copy()

            completion = st.write_stream(stream_tokens(
                st.session_state["pg_model"], 
                messages_to_use, 
                system,
                temperature,
                max_new_tokens))
        st.session_state.messages.append({"role": "assistant", "content": completion})

def rag_chat():
    st.title("RAG Q&A")

    # Process a file.
    with st.expander("Upload, process, and embed a file"):
        pdf_upload = st.file_uploader("Upload file", type=["pdf"])
        chunk_size = st.slider("Chunk size", 10, 3000, 1000)
        chunk_overlap = st.slider("Chunk overlap (number of chars)", 0, 1000, 100)

    with st.expander("Q&A Settings"):
        temperature = st.slider("Temperature", 0.0, 2.0, 0.1)
        max_new_tokens = st.slider("Max new tokens", 1, 2000, 1000)
        model = st.selectbox("Model", models)
        st.session_state["pg_model"] = model        

    if pdf_upload:

        # Process the file
        table = process_pdf(pdf_upload, chunk_size, chunk_overlap)

        # Query the documents
        query_text = st.text_input("Enter your question")

        if st.button("Query"):
            query = embed(query_text)

            st.markdown("## Retrieved results:")
            results = table.search(query).limit(5).to_pandas()
            st.write(results[['text', 'chunk', '_distance']])

            st.markdown("## Answer:")
            answer = gen_tokens(
                st.session_state["pg_model"],
                [{"role": "user", "content": qa_template.format(
                    context=results["text"].tolist()[0], 
                    query=query_text)}],
                "You are a helpful assistant that answers questions based on given context.",
                temperature,
                max_new_tokens
            )
            st.write(answer)

pg = st.navigation([
    st.Page(home, title="Home", icon="🏠"),
    st.Page(llm, title="LLM", icon="🤖"),
    st.Page(lvm, title="LVM", icon="🖼️"),
    st.Page(embedding, title="Embedding", icon="🗜️"),
    #st.Page(guardrails, title="Guardrails", icon="🛡️"),
    st.Page(chat, title="Chat", icon="💬"),
    st.Page(rag_chat, title="RAG Q&A", icon="📄"),
])
pg.run()