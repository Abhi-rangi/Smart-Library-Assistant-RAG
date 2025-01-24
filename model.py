from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
import chainlit as cl
import redis
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
import time
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Load the FAISS vector store from a predefined path.
DB_FAISS_PATH = 'vectorstore/db_faiss'
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
docsearch = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Setup memory and history management for conversation continuity.
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    chat_memory=message_history,
    return_messages=True,
)

# Initialize the ChatGroq model.
llm_groq = ChatGroq(model_name='llama-3.1-70b-versatile')

# Configure the conversational chain with the text data and memory systems.
chain = ConversationalRetrievalChain.from_llm(
    llm=llm_groq,
    retriever=docsearch.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True,
)

global_chain = chain

@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="Welcome to Open Textbook AI Chat Assistant. How may I help you today?", author="Library")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    user_query = message.content.strip().lower()

    cached_response = redis_client.get(user_query)
    if cached_response:
        await cl.Message(content=cached_response, author="Iconcern").send()
        return
        
    await cl.Message(content="Processing your request...", author="Iconcern").send()
    res = await global_chain.ainvoke(user_query+" and also give all information of each book including link, date, and subject if it has")
    answer = res["answer"]

    full_response = answer #+ source_info
    redis_client.set(user_query, full_response,ex=3600)

    await cl.Message(content=full_response, author="Iconcern").send()

