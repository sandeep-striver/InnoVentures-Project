import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from typing import List, Optional
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult
import subprocess

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom LLM class for Ollama that works with LangChain
class OllamaLLM(BaseLLM):
    model_name: str = "llama3"  # Define the model name to use with Ollama

    # Implementing the _call method to execute the model via subprocess
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # Simplifying the command
            clean_prompt = f"Question: {prompt}\nHelpful Answer:"
            result = subprocess.check_output(
                f'ollama generate {self.model_name} --prompt "{clean_prompt}"', 
                shell=True
            ).decode('utf-8')
            
            # Return the cleaned result
            return result.strip()  # Ensure the result is trimmed
        except subprocess.CalledProcessError as e:
            return f"Error: {str(e)}"

    # Implementing the _generate method required by LangChain
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            result = self._call(prompt, stop)
            generations.append([{"text": result}])
        
        # Return the generated result in LangChain's expected format
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "ollama_llm"

# Streamlit App Title and Subtitle
st.title("Chat with CSV using Llama3 🦙🦜 via Ollama")
st.markdown("<h3 style='text-align: center; color: white;'>Built by InnoVentures</h3>", unsafe_allow_html=True)

# File uploader in Streamlit sidebar
uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    # Temporary file to store uploaded CSV
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load the CSV file and embed its data
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    # Initialize custom Ollama LLM
    llm = OllamaLLM()  # Use the custom Ollama LLM
    
    # Create a ConversationalRetrievalChain using the Ollama LLM and FAISS retriever
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # Function to handle the conversational chat with the CSV data
    def conversational_chat(query):
        # Get the answer from the retrieval chain and clean it up
        result = chain({"question": query, "chat_history": st.session_state['history']})
        
        # Ensure the answer returned is valid and trimmed
        answer = result["answer"].strip() if "answer" in result else "I don't know the answer."
        
        # Update the session state with the new query and answer
        st.session_state['history'].append((query, answer))
        return answer
    
    # Initialize session state for chat history and past interactions
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    # Containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            # Get the output for the user's query
            output = conversational_chat(user_input)
            
            # Update session state with user input and generated response
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Displaying the conversation history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

