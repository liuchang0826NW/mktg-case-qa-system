from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from utils import OPEN_API_KEY, PINECONE_API_KEY, PINECONE_ENV


def load_docs():
    loader = UnstructuredPDFLoader("/Users/changliu/CL_Workspace/qa/mktg-cases.pdf")
    data = loader.load()
    print(f'You have {len(data)} documents in your data')
    print(f'There are {len(data[0].page_content)} characters in your document')
    return data


documents = load_docs()


def split_docs(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


docs = split_docs(documents)
print(len(docs))
print(docs[10])

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",
                              openai_api_key=OPEN_API_KEY)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = "acliu-mktg-cases"
index = Pinecone.from_texts([doc.page_content for doc in docs], embeddings, index_name=index_name)













