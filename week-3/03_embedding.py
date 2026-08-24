from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

documents = [
    "Mobile phones comes with 2 OS android and iOS.",
    "Android phones built by many companies like Samsung, OnePlus, Xiaomi, and Google.",
    "iOS Made exclusively by Apple.",
    "Android Lets you change home screens, use custom widgets, and alter how the system looks."
    "Android Often use advanced AI tools, fast battery charging, and different hardware styles like folding screens.",
    "iOS Offers a simple, clean, and uniform layout that is easy for anyone to learn.",
    "iOS  Works seamlessly with Mac computers, iPads, and Apple Watches using shared clips and messaging.",
    "Latest Android version is 17.",
    "Latest iOS version is 26.0."
]

vector_store = FAISS.from_texts(
    documents,
    embedding=embeddings
)

query = input("Ask mobile related question: ")

results = vector_store.similarity_search(
    query,
)

print("\nQuery:")
print(query)

print(f"\nMost similar documents: \n{results[0].page_content}")

# for index, document in enumerate(results, start=1):
#     print(f"\n{index}. {document.page_content}")