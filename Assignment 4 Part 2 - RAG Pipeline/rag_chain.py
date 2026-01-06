import os
from dotenv import load_dotenv
from typing import Any
from langchain_community.llms import Ollama
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import Runnable
from gradio_client import Client
from pydantic import PrivateAttr
from langchain_core.outputs import Generation, LLMResult
from retriever import Retriever
from ta_pipeline import get_llm_port

class GradioLLMWrapper(BaseLLM, Runnable):
    # NOTE : This class is given and does not have to be implemented or changed.
    _client: Any = PrivateAttr()

    def __init__(self, space_name: str, hf_token: str):
        super().__init__()
        object.__setattr__(self, "_client", Client(space_name, hf_token=hf_token))

    def _call(self, prompt: str, **kwargs: Any) -> str:
        result = object.__getattribute__(self, "_client").predict(prompt, api_name="/predict")
        return result

    def invoke(self, input: str, **kwargs: Any) -> str:
        return self._call(input, **kwargs)

    def _generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        return [self._call(prompt, **kwargs) for prompt in prompts]

    def generate(self, prompts: list[str], **kwargs: Any) -> LLMResult:
        generations = self._generate(prompts, **kwargs)
        return LLMResult(
            generations=[[Generation(text=gen)] for gen in generations]
        )

    @property
    def _llm_type(self) -> str:
        return "gradio-flan"

class RAG_Chain:
    def __init__(self, data_dir, llm_type="gradio_flan", init_retriever=True, llm_model="llama3.2", llm_ag=None): 
        '''
        Initializes the RAG chain by selecting an LLM backend and loading the document retriever system.
        Load the api-key & space name from the .env file and initialize self.llm and self.retriever_system
        You will need to get an API Access Token to access the HuggingFace models. See notebook
        instructions in section 7.2.0 for more information on obtaining an API Access Token, space name, and setting up 
        a .env file to store the token. DO NOT upload your API token to Gradescope.

        Args:
            data_dir: String path of folder location of PDFs to load
            llm_ag: Custom LLM agent for injecting a pre-initialized model manually.
            llm_type: String to determine which llm to use (for Q8)
            init_retriever: boolean switch on whether to instantiate retriever (for Q8)
            llm_model: String specifiying which ollama model to use (for Q8)

        Initialize:
            self.llm: LLM from HuggingFaceHub. 
            self.retriever_system: your implemented retrieval system from retriever.py

        Returns:
            None

        NOTE : This function is given and does not have to be implemented.
        '''

        if llm_ag is not None:
            self.llm = llm_ag

        elif llm_type == "ollama_only":
            self.set_ollama_only(llm_model=llm_model)

        elif llm_type == "flask_ollama":
            self.set_flask_ollama(llm_model=llm_model)

        elif llm_type == "gradio_flan":
            load_dotenv()
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            space_name = os.getenv("GRADIO_SPACE_NAME") 
            self.llm = GradioLLMWrapper(
                space_name="OMalcolm1/NLP_HW4_Part2",
                hf_token=api_key
            )


        # load the retriever system - do not change
        if init_retriever:
            self.retriever_system = self.init_retriever_system(data_dir)

    def set_ollama_only(self, llm_model="llama3.2"):
        '''
        Q8: Initialize self.llm using Ollama and set model to the llm_model

        Args:
            llm_model: String specifiying which ollama model to use

        Initialize:
            self.llm: llm_model from ollama

        Returns:
            None
        Hint: See the Jupyter Notebook for documentation on the imported Ollama wrapper
        '''
        self.llm = Ollama(model=llm_model)            # Replace None with Ollama
        
    def set_flask_ollama(self, llm_model="llama3.2", api_key="None"):
        '''
        Q8: Initialize self.llm using the OpenAI wrapper and set model to the llm_model. 
        The required fields are openai_api_base, openai_api_key, and model_name.
        Note that our implementation does not require a key, but this is a required field and needs a placeholder.

        Args:
            llm_model: String specifiying which ollama model to use

        Initialize:
            self.llm: llm_model via Flask Ollama

        Returns:
            None
        Hint: get_llm_port from the TA Pipeline may be helpful
        '''

        # load the LLM
        port = get_llm_port()
        self.llm = OpenAI(
            model_name = llm_model,
            openai_api_base = f"http://localhost:{port}/v1",
            openai_api_key = api_key
        )           # Replace None with Flask Ollama

    def query_the_llm(self, question):
        """
        Invokes a question to the RAG's LLM without any supporting documents.
        NOTE : This function is given and does not have to be implemented.
        """
        response = self.llm.invoke(question)
        return response

    def init_retriever_system(self, data_dir):
        '''
        Initialize your retriever system by instantiating your retriever implementation from retriever.py and loading the documents in the PDF directory.
        Split the loaded documents into chunks and use the chunks to create and return the VectorStoreRetriever

        Args:
            data_dir: String path of folder location of PDFs to load

        Returns:
            retriever: langchain VectorStoreRetriever

        HINT: You can refer to the local test code in 7.1.3 to see an example of how we initialize a retriever.
        You may omit chunk_size, chunk_overlap, and num_chunks_to_return to use their default values.
        '''
        retriever = Retriever()
        documents = retriever.loadDocuments(data_dir)
        chunks = retriever.splitDocuments(documents)
        return retriever.createRetriever(chunks)

    def createPrompt(self, question):
        '''
        Define the prompt template and return a formatted prompt using the template and question argument.

        Args:
            question: Dictionary with the following keys: 'question', 'A', 'B', 'C', 'D'. See notebook for example
        
        Returns:
            formatted_prompt: The question and answer choices reformatted using the prompt template to use to query the LLM.
        '''
        template = ChatPromptTemplate.from_template(
            "You are a helpful assistant answering multiple choice questions.\n\n"
            "Question: {question}\n\n"
            "A. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
            "Please pick the best answer (A, B, C, or D) and explain your reasoning."
        )

        return template.format(**question)

    def createRAGChain(self):
        '''
        Build the RAG pipeline using the RetrievalQA chain. Make sure to pass the LLM (self.llm) and retriever system (self.retriever_system).
        Hint: https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html#langchain.chains.retrieval_qa.base.RetrievalQA.from_chain_type

        Args:
            None
            
        Returns:
            qa_chain: BaseRetrievalQA used to answer multiple choice questions.
        '''
        qa_chain = RetrievalQA.from_chain_type(
        llm=self.llm,
        retriever=self.retriever_system,
        return_source_documents=False
        )
        return qa_chain
