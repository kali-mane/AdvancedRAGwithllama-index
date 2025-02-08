from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, SummaryIndex, Settings
from llama_index.core.objects import ObjectIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from config import Config


Config.set_api_key()

def get_index(doc):
    index = None
    documents = PDFReader().load_data(file=doc)
    # wealth_docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    summary_index = SummaryIndex(nodes, storage_context=storage_context)
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    return summary_index, vector_index
