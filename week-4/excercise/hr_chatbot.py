import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)

import os
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain_core.documents import Document

from langchain_community.document_loaders import PDFPlumberLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_mistralai import MistralAIEmbeddings
from langchain_groq import ChatGroq

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class HRChatbot:

    def __init__(self):
        documents = self.load_document()

        chunks = self.split_document(documents)

        embeddings = self.create_embeddings()

        self.vector_store = self.create_vector_store(
            chunks,
            embeddings
        )

        self.retriever = self.create_retriever(
            self.vector_store
        )

        self.model = self.create_model()

        self.prompt = self.create_prompt()

    def load_document0(self):
        pdf_path = "../documents/hr_policy.pdf"
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()
        return documents    

    def load_document(self):
        pdf_path = "../documents/hr_policy.pdf"

        reader = PdfReader(pdf_path)
        documents = []
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            metadata = {
                "source": pdf_path,
                "page": page_number
            }
            document = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(document)
        return documents

        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()
        return documents    

    def split_document(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(documents)
        return chunks

    def create_embeddings(self):
        embeddings = MistralAIEmbeddings(
            model="mistral-embed"
        )
        return embeddings

    def create_vector_store(self, chunks, embeddings):
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        return vector_store

    def create_retriever(self, vector_store):
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        return retriever

    def create_model(self):
        model = ChatGroq(
            model=os.environ.get("GROQ_MODEL_OPEN"),
            temperature=0
        )
        return model

    def create_context(self, documents):
        context = "\n\n".join(
            document.page_content
            for document, score in documents
        )   
        return context

    def search_documents(self, question, k=3):
        results = self.vector_store.similarity_search_with_score(
            question,
            k=k
        )
        return results

    def retrieve_documents(self, question):
        documents = self.retriever.invoke(
            question
        )
        return documents

    def create_prompt(self):
        return ChatPromptTemplate.from_messages([
            (
                "system",
                """
                    You are an HR policy assistant.

                    Answer the user's question using ONLY the provided HR policy context.

                    Rules:
                        - Do not use outside knowledge.
                        - Do not make up information.
                        - If the answer cannot be found in the context, say:
                            "I could not find this information in the HR policy."
                        - Keep the answer clear and concise.

                    HR Policy Context:
                    {context}
                """
            ),
            (
                "human",
                "{question}"
            )
        ]
    )

    def generate_answer(
        self,
        context,
        question
        ):
            messages = self.prompt.invoke({
                "context": context,
                "question": question
            })
            response = self.model.invoke(messages)
            return response.content

    def chat(self, question):
        results = self.search_documents(question)
        context = self.create_context(results)
        answer = self.generate_answer(
            context,
            question
        )

        sources = []
        for document, score in results[:1]:
            sources.append({
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page")+1,
                "score": float(score)
            })

        return {
            "answer": answer,
            "sources": sources
        }



if __name__ == "__main__":

    chatbot = HRChatbot()

    while True:
        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = chatbot.chat(question)

        print("\nAnswer:")
        print(response["answer"])
        print("\nSources:")
        
        for index, source in enumerate(
            response["sources"],
            start=1
        ):
            print(f"Source : {source['source']}")
            print(f"Page   : {source['page']}")
            print(f"Score  : {source['score']}")




# What is the name of the company?
# How many earned leaves granted per year?
# How many daily average required to count complete working day?
# What is the notice period for resignation? 
# How many days of sick leaves granted per year?
# What is the charges in case of punching card is lost?
# How many days of marriage leave are employees eligible for?
# What is the reimbursement amount per person for dinner/lunch? 
# How many National/Festival holidays are declared per year?
# An employee takes 20 continuous working days of leave. By how many months will their appraisal cycle be extended, and does this differ if they're still on probation vs. confirmed?
#  If an employee takes a planned leave on Friday, can they also take a sick leave on the following Monday? Why or why not?