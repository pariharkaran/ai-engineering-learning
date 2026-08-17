import os 

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model=os.environ.get("GROQ_MODEL"),
    temperature=0.9,
    model_kwargs={"top_p": 0.1},
    max_tokens=1000
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains the meaning of terms in one sentence."),
    ("human", "Explain Meaning of {name} in one sentence")
])

input = input("Enter a term to understand via ai: ")

# formatted_prompt = prompt.invoke({"name": input})
# response = model.invoke(formatted_prompt)
 
chain = prompt | model | StrOutputParser()

response = chain.invoke({"name": input})

print(f"\nresponse : \n{response}\n")