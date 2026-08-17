import os

from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class Technology(BaseModel):
    success:bool
    name: str | None = None
    category:str | None = None
    difficulty:str | None = None
    message:str

class Expense(BaseModel):
    restaurant_name:str
    location:str
    order:str
    amount:int

class Country(BaseModel):
    language:str
    religion:str
    name:str

model = ChatGroq(
    model=os.environ.get("GROQ_MODEL"),
    temperature=1
)

structured_model = model.with_structured_output(Technology)

prompt_template = ChatPromptTemplate.from_messages([
 (
        "system",
        """
        Determine whether the user's request is related to a technology.

        If it is related to a technology:
        - success = true
        - provide name, category and difficulty
        - provide a helpful message

        If it is NOT related to a technology:
        - success = false
        - name = null
        - category = null
        - difficulty = null
        - explain why in message

        Always return the required structured output.
        """
    ),
    ("human","Explain the technical terms asked by the user: {user_prompt}")
])

chain = prompt_template | structured_model

user_prompt = input("Ask your ai about anything and get structured output: ")

try:
    response = chain.invoke({"user_prompt":user_prompt})
    print(f"\n response: \n",response)
except Exception as e:
    print(f"Error: ",e)


