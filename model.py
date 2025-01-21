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

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def similarity_score(text1, text2):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    return cosine_score

# Establish a connection to the Redis server for caching responses.
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
    retriever=docsearch.as_retriever(search_kwargs={"k": 10}),
    memory=memory,
    return_source_documents=True,
)

global_chain = chain

# Load coherence evaluation model
coherence_evaluator = pipeline("text-generation", model="gpt2")

@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="Welcome to Open Educational Resources, your chat assistant. How may I help you today?", author="Library")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    user_query = message.content.strip().lower()
    start_time = time.time()

    cached_response = redis_client.get(user_query+" and also give all information of each book including link, date, and subject if it has")
    if cached_response:
        await cl.Message(content=cached_response, author="Iconcern").send()
        return

    await cl.Message(content="Processing your request...", author="Iconcern").send()
    res = await global_chain.ainvoke(user_query+" and also give all information of each book including link, date, and subject if it has")
    answer = res["answer"]
    source_documents = res["source_documents"]
    processing_time = time.time() - start_time

    # Evaluate response coherence
    reference = coherence_evaluator(user_query, num_return_sequences=1)[0]['generated_text']
    coherence_score = similarity_score(answer, reference)  # Define your similarity function

    text_elements = []
    source_info = ""
    relevant_docs_count = 0

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         text_elements.append(cl.Text(content=source_doc.page_content, name=source_name))
    #         if is_relevant(source_doc.page_content, user_query):  # Define your relevance function
    #             relevant_docs_count += 1
    #     source_names = [text_el.name for text_el in text_elements]
        # source_info = f"\nSources: {', '.join(source_names)}" if source_names else "\nNo sources found"

    # retrieval_accuracy = (relevant_docs_count / len(source_documents)) * 100 if source_documents else 0

    full_response = answer + source_info
    redis_client.set(user_query, full_response)

    # Log performance metrics
    print(f"Latency: {processing_time:.2f} seconds")
    # print(f"Retrieval Accuracy: {retrieval_accuracy}%")
    print(f"Response Coherence Score: {coherence_score}")

    # Send the final response to the user
    await cl.Message(content=full_response, author="Iconcern", elements=text_elements).send()

# # Start the chat server
# cl.run()
def is_relevant(document_content, query):
    doc_embedding = model.encode(document_content, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(doc_embedding, query_embedding).item()
    return similarity > 0.5  # Consider as relevant if similarity is above 0.5
