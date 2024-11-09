from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import streamlit as st
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import numpy as np

# Define custom functions for processing text and documents
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=10, length_function=len, add_start_index=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.add_start_index = add_start_index

    def split_text(self, text):
        chunks = []
        start_index = 0
        while start_index + self.chunk_size <= len(text):
            chunk = text[start_index:start_index + self.chunk_size]
            if self.add_start_index:
                chunk = f"{start_index}:{chunk}"
            chunks.append(chunk)
            start_index += self.chunk_size - self.chunk_overlap
        if start_index < len(text):
            chunks.append(text[start_index:])
        return chunks

class BartEmbeddings:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_documents(self, documents):
        embeddings = []
        for doc in documents:
            # Iterate over document pages for loaders like PyPDFLoader
            for page in doc.pages:
                # Process each page content
                page_content = page.content
                # Split text into chunks if necessary
                if len(page_content) > 1000:
                    page_content = RecursiveCharacterTextSplitter.split_text(page_content)[0]
                # Encode and embed the chunk
                inputs = self.tokenizer(page_content, return_tensors="pt")
                outputs = self.model(**inputs)
                flat_list = [item for sublist in outputs.logits.mean(dim=1).detach().numpy().tolist() for item in sublist]
                embeddings.append(flat_list)
        return embeddings

# Define your chosen document analysis logic here
def analyze_documents(documents, query):
    key_points = []
    passages = []
    document_sentiments = {}

    # Loop through each document
    for document in documents:
        document_id = document.identifier
        document_sentiments[document_id] = {}

        # Extract key points using your chosen summarization technique
        document_summary = summarize_document(document.content)
        key_points.append(document_summary.split(". ")[:3])  # Extract first 3 sentences as key points

        # Find relevant passages supporting the query using your retrieval method
        relevant_passages = pdf_qa.retrieve(query, document_id, k=3)  # Adjust search parameters

        # Analyze sentiment of each passage and the document itself
        for passage in relevant_passages:
            passage_sentiment = TextBlob(passage.snippet).sentiment
            document_sentiments[document_id]["passages"][passage.identifier] = passage_sentiment
        document_sentiments[document_id]["document"] = TextBlob(document.content).sentiment

        # Collect relevant passages with their sentiments
        passages.append({passage.identifier: {"snippet": passage.snippet, "sentiment": passage_sentiment}})

    return key_points, passages, document_sentiments

def summarize_document(document_text):
    # Use sentence transformers to encode document sentences
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    sentences = document_text.split(". ")
    sentence_embeddings = model.encode(sentences)

    # Choose appropriate method to select key sentences (e.g., top-k based on cosine similarity)
    top_k_sentences = sorted(enumerate(sentence_embeddings), key=lambda x: x[1], reverse=True)[:3]  # Select top 3 sentences
    key_sentences = [sentences[i] for i, _ in top_k_sentences]

    # Combine key sentences into a summary
    document_summary = ". ".join(key_sentences)
    return document_summary


def embed_documents(self, documents):
    embeddings = []
    for doc in documents:
        # Iterate over document pages for loaders like PyPDFLoader
        for page in doc.pages:
            # Process each page content
            page_content = page.content
            # Split text into chunks if necessary
            if len(page_content) > 1000:
                page_content = RecursiveCharacterTextSplitter.split_text(page_content)[0]
            # Encode and embed the chunk
            inputs = self.tokenizer(page_content, return_tensors="pt")
            outputs = self.model(**inputs)
            flat_list = [item for sublist in outputs.logits.mean(dim=1).detach().numpy().tolist() for item in sublist]
            embeddings.append(flat_list)
    return embeddings

# Define document embedding logic 
def document_embeddings(self, document):
    sentence_embeddings = []
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    for page in document.pages:
        sentences = page.content.split(". ")
        sentence_embeddings.extend(model.encode(sentences))
    # Average or pool sentence embeddings to represent the document
    document_embedding = np.mean(sentence_embeddings, axis=0)
    return document_embedding

# Process uploaded files
def process_uploaded_file(file_path):
    try:
        if file_path.endswith(".pdf"):
            document = PyPDFLoader(file_path).load()
        elif file_path.endswith(".docx"):
            document = Docx2txtLoader(file_path).load()
        else:
            document = TextLoader(file_path).load()
        document_embeddings = document_embeddings(document)  # Use your chosen document embedding logic
        return document, document_embeddings
    except Exception as e:
        st.error(f"Error processing file '{file_path}': {e}")
        return None, None

# UI and document management
documents = []
embeddings = []
documents_changed = False
text_splitter = RecursiveCharacterTextSplitter()

def document_loaded():
    return vectordb is not None

# LLM interaction logic (replace with your desired implementation)
def chat_with_llm(query):
    # Use pdf_qa to retrieve relevant documents based on the query
    relevant_docs = pdf_qa.retrieve(query)

    # Analyze documents and query for key points, supporting passages, and sentiment
    key_points, passages, document_sentiments = analyze_documents(relevant_docs, query)  # Replace with your chosen analysis function
    query_sentiment = TextBlob(query).sentiment

    # Craft a response referencing key points, highlighting passages, adjusting tone based on sentiment, and personalizing based on user preferences (if available)
    llm_response = f"Regarding your question about {query}, I found some relevant documents. Here are some key points:"
    for point, passage in zip(key_points, passages):
        llm_response += f"\n - {point}. This is supported by the following passage from document {passage.identifier}: {passage.snippet}"

    # Adjust tone based on sentiment
    positive_documents = [doc_id for doc_id, sentiment in document_sentiments.items() if sentiment["positive"] > sentiment["negative"]]
    negative_documents = [doc_id for doc_id, sentiment in document_sentiments.items() if sentiment["negative"] > sentiment["positive"]]

    if query_sentiment.positive > query_sentiment.negative:
        llm_response += ". These documents seem informative. Would you like me to focus on ..."
    elif query_sentiment.negative > query_sentiment.positive:
        llm_response += ". I understand your concern. Some documents seem critical, but perhaps ..."
    elif positive_documents:
        llm_response += ". These documents offer positive perspectives on ... Would you like to know more?"
    elif negative_documents:
        llm_response += ". These documents raise concerns about ... Would you like me to elaborate?"
    else:
        llm_response += "."

    # Encourage further dialogue
    if len(key_points) > 1:
        llm_response += ". Which aspect of these points would you like to know more about?"
    elif len(key_points) == 1:
        llm_response += ". Is there anything specific you'd like to know about this point?"
    return llm_response, relevant_docs

# Main loop
document_loaded = False  # Initial state flag
afile = 904385093284052
st.title("Chat with your Documents")
st.write("Ask me anything about the documents you uploaded!")
# Upload documents
uploaded_files = st.file_uploader("Upload text files:", accept_multiple_files=True,key=f"{afile}")

# Check if any documents were uploaded
if uploaded_files:
    documents_changed = True

    # Process uploaded files and initialize models (only if documents changed)
    if not document_loaded:
        documents = []
        embeddings = []
        for file in uploaded_files:
            document, document_embeddings = process_uploaded_file(file.name)
            if document:
                documents.append(document)
                embeddings.extend(document_embeddings)

        # Initialize models and vector store (only once)
        try:
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
            model = AutoModel.from_pretrained("facebook/bart-base")
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=BartEmbeddings(model, tokenizer),
                persist_directory="./data")
            vectordb.persist()
            pdf_qa = ConversationalRetrievalChain.from_llm(
                bart_embeddings=BartEmbeddings(model, tokenizer),
                retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
                return_source_documents=True,
                verbose=False
            )
            document_loaded = True  # Update state flag
        except Exception as e:
            st.error(f"Error initializing models and vector store: {e}")
            documents = []
            embeddings = []
            documents_changed = False

    # Chat interaction only if document loaded
    if document_loaded:
        # User query and chat interaction
        query = st.text_input("Ask your question:", value="")
        if st.button("Chat"):
            response, relevant_docs = chat_with_llm(query)
            st.markdown(f"**LLM Response:** {response}")
            if relevant_docs:
                st.markdown("**Relevant Documents:**")
                for doc in relevant_docs:
                    st.markdown(f"**{doc.identifier}:** {doc.snippet}")

            # Reset flags and input field for next iteration
            st.button("Clear")
            documents_changed = False

st.markdown("---")) 
