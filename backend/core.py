import os
from typing import Any

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# from langchain.vectorstores import Pinecone
from langchain_pinecone import Pinecone
import pinecone


from consts import INDEX_NAME

# pinecone.init(
#     api_key=os.environ["PINECONE_API_KEY"],
#     environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
# )


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})


if __name__ == "__main__":
    print(
        run_llm(query="What is LangChain?")
    )  # got response:{'query': 'What is LangChain?', 'result': 'LangChain is a framework that provides tools for working with natural language processing tasks, such as creating structured outputs, parsing text, and generating responses.
    # It includes components like chains for running sequences of tasks, integration with OpenAI tools, and support for Pydantic schemas for defining data models.',
    # 'source_documents': [Document(metadata={'source': 'https://https://api.python.langchain.com/en/latest/community_api_reference.html'}, page_content='from the serialized     field. :return: The modified list of runs.'),
    # Document(metadata={'source': 'https://https://api.python.langchain.com/en/latest/chains/langchain.chains.structured_output.base.create_structured_output_runnable.html'}, page_content='Returns\nA runnable sequence that will return a structured output(s) matching the givenoutput_schema.\nReturn type\nRunnable\nOpenAI tools example with Pydantic schema (mode=’openai-tools’):from typing import Optional\nfrom langchain.chains import create_structured_output_runnable\nfrom langchain_openai import ChatOpenAI\nfrom langchain_core.pydantic_v1 import BaseModel, Field\nclass RecordDog(BaseModel):\n    \'\'\'Record some identifying information about a dog.\'\'\'\n    name: str = Field(..., description="The dog\'s name")\n    color: str = Field(..., description="The dog\'s color")'),
    # Document(metadata={'source': 'https://https://api.python.langchain.com/en/latest/agents/langchain.agents.format_scratchpad.log_to_messages.format_log_to_messages.html'}, page_content='Returns\nThe scratchpad.\nReturn type\nList[BaseMessage]'),
    # Document(metadata={'source': 'https://https://api.python.langchain.com/en/latest/output_parsers/langchain.output_parsers.retry.RetryWithErrorOutputParser.html'}, page_content='The return value is parsed from only the first Generation in the result, whichis assumed to be the highest-likelihood Generation.\nParameters\nresult (List[Generation]) – A list of Generations to be parsed. The Generations are assumed\nto be different candidate outputs for a single model input.\npartial (bool) – \nReturns\nStructured output.\nReturn type\nT\nasync aparse_with_prompt(completion: str, prompt_value: PromptValue) → T[source]¶\nParameters\ncompletion (str) – \nprompt_value (PromptValue) – \nReturn type\nT')]}
