import gradio as gr
from utils import OPEN_API_KEY, PINECONE_API_KEY, PINECONE_ENV
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import pinecone


llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPEN_API_KEY)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPEN_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index = Pinecone.from_existing_index(index_name="acliu-mktg-cases", embedding=embeddings)

chain = load_qa_chain(llm, chain_type="stuff")

def respond(question):
  similar_docs = index.similarity_search(question, k=20)
  return chain.run(input_documents=similar_docs, question=question)

inputs = gr.Textbox(label="MKTG Cases QA System based on lang chain")
outputs = gr.TextArea(label="Reply")
gr.Interface(fn=respond, inputs=inputs, outputs=outputs, title="Ask me!",
             description="Ask about MKTG cases").launch()
