import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core import Settings
import google.generativeai as genai
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.llms.gemini import Gemini


class Config:

    @classmethod
    def set_api_key(cls):
        api_key = load_dotenv(find_dotenv())
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)
        Settings.embed_model = GeminiEmbedding(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)
        Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
