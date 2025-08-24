import os
import dotenv as load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# load environment variable from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key, 
                                                        model_name= "text-embedding-3-small")

# Initialize the chroma client with persistance
