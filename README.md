# InnoVentures

Chat with CSV using Llama3 via Ollama.
This Streamlit application allows users to interact with a CSV file using a conversational AI model, Llama3, 
via Ollama. The app facilitates querying data from the CSV file and retrieving answers through a chat interface.


![Screenshot 2024-09-15 150923](https://github.com/user-attachments/assets/9b62a89a-58a0-40c7-979a-519f4eac78d9)



TECHSTACK  Used in It

 *Python 3.12.4

 *Streamlit

 -streamlit_chat library


 -LangChain

 -Ollama CLI

 -sentence-transformers library

 -FAISS library



 * Code Explanation

OllamaLLM Class: A custom class that integrates the Ollama language model with LangChain. It uses subprocess to interact with the Ollama CLI for generating responses.


Streamlit App:



The title and Subtitle are Set up at the Streamlit interface.

File Uploader: Allows users to upload a CSV file.

Reads the CSV, embeds its content using HuggingFaceEmbeddings, and stores it in a FAISS vector store.

Uses a ConversationalRetrievalChain to handle user queries and retrieve relevant data from the CSV.

Manages user input and conversation history, displaying chat messages through Streamlit.
