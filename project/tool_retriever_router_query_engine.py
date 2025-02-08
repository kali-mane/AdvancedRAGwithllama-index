from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
from note_engine import notes_engine
from config import Config

Config.set_api_key()

def tool_retriever_router_query_engine(summary_index, vector_index):
    summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")
    vector_query_engine = vector_index.as_query_engine()

    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_engine",
                description=(
                    "useful for retrieving specific context from the agriculture document"
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_engine",
                description=(
                    "useful for summarization questions related to agriculture document"
                ),
            ),
        ),
    ]

    #query_engine_tools = pdf_tools()
    obj_index = ObjectIndex.from_objects(
        query_engine_tools, index_cls=VectorStoreIndex
    )

    router_query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever())
    return router_query_engine