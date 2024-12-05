import requests
import json
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import chromadb

# Function to call Ollama API for chat responses
def call_llm_api(messages):
    url = "http://36.50.40.36:11435/api/chat"
    data = {
        "model": "llama3.2:latest",
        "messages": messages,
        "stream": True 
    }

    response = requests.post(url, json=data, stream=True)
    if response.status_code == 200:
        print("Streaming output:")
        
        # Process each line in the response as it arrives
        for line in response.iter_lines():
            if line:  # Ensure line is not empty
                try:
                    # Parse each line as JSON
                    json_line = json.loads(line.decode('utf-8'))  
                    # Extract and print the content if it exists
                    if 'message' in json_line and 'content' in json_line['message']:
                        print(json_line['message']['content'], end=' ', flush=True)  
                except json.JSONDecodeError:
                    print("Failed to decode JSON:", line)
    else:
        print("Error:", response.status_code, response.text)

    # if response.status_code == 200:
    #     output = ""
    #     for line in response.iter_lines():
    #         if line:
    #             try:
    #                 json_line = json.loads(line.decode('utf-8'))
    #                 if 'message' in json_line and 'content' in json_line['message']:
    #                     output += json_line['message']['content'] + " "
    #             except json.JSONDecodeError:
    #                 print("Failed to decode JSON:", line)
    #     return output.strip()
    # else:
    #     print("Error:", response.status_code, response.text)
    #     return None


def get_embeddings(texts):
    url = "http://36.50.40.36:11435/api/embeddings"
    embeddings = []

    for text in texts:
        data = {
            # "model": "nomic-embed-text",
            "model": "all-minilm:33m",
            "prompt": text
        }
        
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            embedding_response = response.json()
            if 'embedding' in embedding_response and embedding_response['embedding']:
                embeddings.append(embedding_response['embedding'])
            else:
                print(f"No embedding generated for '{text}':", embedding_response)
                embeddings.append([])  
        else:
            print("Error:", response.status_code, response.text)
            embeddings.append([]) 
            
    return embeddings

# Load content from the web
loader = WebBaseLoader(
   "https://www.daraz.com.bd/products/beauty-glazed-nose-pore-strips-blackhead-remover-5-pcs-i314756792-s1418019765.html?pvid=2c874dd2-4db0-4fb8-b335-60bb58d67c38&search=jfy&scm=1007.28811.376629.0&priceCompare=skuId%3A1418019765%3Bsource%3Atpp-recommend-plugin-41701%3Bsn%3A2c874dd2-4db0-4fb8-b335-60bb58d67c38%3BunionTrace%3A2102fc9a17284642427386902e2d58%3BoriginPrice%3A9300%3BvoucherPrice%3A9300%3BdisplayPrice%3A9300%3BsourceTag%3A%23auto_collect%231%24auto_collect%24%3BsinglePromotionId%3A50000023256376%3BsingleToolCode%3AflashSale%3BvoucherPricePlugin%3A1%3BbuyerId%3A0%3ButdId%3A-1%3Btimestamp%3A1728464242902&spm=a2a0e.tm80335411.just4u.d_314756792"
)
docs = loader.load()
print(f"Loaded {len(docs)} documents")
# print(f"Document content: {docs}")

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Reduced chunk size
splits = text_splitter.split_documents(docs)
print(f"Generated {len(splits)} document splits")

# Generate unique IDs for each document chunk if splits are available
ids = [str(i) for i in range(len(splits))]  # Ensure IDs are always defined

if len(splits) > 0:
    print(f"Generated {len(ids)} unique IDs")
else:
    print("Error: No document splits available.")

class CustomEmbeddingFunction:
    def embed_documents(self, texts):
        return get_embeddings(texts)
    
    def embed_query(self, query):
        return get_embeddings([query])[0]

# Use the custom embedding function with Chroma
embedding_function = CustomEmbeddingFunction()

collection_name = "my_collection"
persist_directory = "./chroma_db"
# Ensure splits and ids are populated before initializing the vector store
if len(splits) == len(ids) and len(splits) > 0:
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function, ids=ids,persist_directory=persist_directory)
    print("Vectorstore successfully initialized.")
 
else:
    print("Error: Mismatch between document splits and IDs, or no splits available.")

# Retrieve context from the vector store if successfully initialized
#vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

retriever = vectorstore.as_retriever()
user_question = "give me product's price in ৳ and  top 5 review detals and overall review in english"
context_docs = retriever.invoke(input=user_question)

print(f"Context docs: {context_docs}")


# Format documents for context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

formatted_context = format_docs(context_docs)

# Prepare messages for Ollama 
messages = [
    
        {"role": "system", "content": "You are an assistant designed to provide helpful responses."},
        {"role": "user", "content": f"Context: {formatted_context}\nQuestion: {user_question}"}
    ]


    # Call the Ollama API and get the response
response_text = call_llm_api(messages)


# Integrate with RAG chain (if needed)
# prompt = hub.pull("rlm/rag-prompt")
# rag_chain = (
#     {"context": formatted_context, "question": RunnablePassthrough()}
#     | prompt
#     | StrOutputParser()
# )

