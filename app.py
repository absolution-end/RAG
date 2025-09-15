# Load API
# Set up local embedding with "SentenceTransformer"
# Create a persistant Chroma collection to store and search docs.
# connect to OpenRouter
# Ask a model a question
# print the answer

import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# load environment variable from .env file
load_dotenv()
os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"
openrouter_key = os.getenv("OPENROUTER_API_KEY")

local_ef = embedding_functions.SentenceTransformerEmbeddingFunction( 
                                                        model_name= "all-MiniLM-L6-v2")

# Initialize the chroma client with persistance
chroma_client = chromadb.PersistentClient(path="chroma_storage")
collection_name = "Doc_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name ,
    embedding_function=local_ef
    )

client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")

resp = client.chat.completions.create(
    model="mistralai/mistral-7b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is human life expectancy in the United States?",
        },
    ],
)
print(resp.choices[0].message.content)