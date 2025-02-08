import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core import Settings
import google.generativeai as genai
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from embeddings import get_index
from tool_retriever_router_query_engine import tool_retriever_router_query_engine
from note_engine import save_note
from config import Config

Config.set_api_key()

document = os.path.join("../data", "Agricultural Produce (Grading and Marking) Act, 1937 _ Directora.pdf")

summary_index, vector_index = get_index(document)

query_engine = tool_retriever_router_query_engine(summary_index, vector_index)
query = "What is the summary of the document"
print(query)
response = query_engine.query(query)
print(response)