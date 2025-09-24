import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq   # 🔹 Groq integration

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # make sure your .env has GROQ_API_KEY

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load documents
text_loader = TextLoader("TMM.txt", encoding="utf-8")
text_docs = text_loader.load()

pdf_loader = PyPDFLoader("faq.pdf")
pdf_docs = pdf_loader.load()

# Combine documents
docs = text_docs + pdf_docs

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(docs)

# Create embeddings & vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 🔹 Define LLM with Groq
# Instead of model="llama3-8b-8192", use:
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"  # or "llama-3.3-70b-versatile", whichever fits your use case
)


# Prompt template with context
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant.\n"
     "- If the user’s question matches the context below, answer using ONLY that context.\n"
     "- If the question is casual (like greetings, thanks, etc.) or not covered in context, "
     "respond naturally as a friendly assistant.\n\n"
     "Context:\n{context}"),
    ("user", "{question}")
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# API route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"reply": "Please provide a message."})

    try:
        docs = retriever.get_relevant_documents(user_message)
        context = "\n\n".join([doc.page_content for doc in docs])

        if not context.strip():  
               # No relevant context → let the LLM handle it normally
               response = llm.invoke([{"role": "user", "content": user_message}])
        else:
               # With context → restrict
               response = chain.invoke({"context": context, "question": user_message})


        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
