from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


store = Chroma.from_texts(
    ["Employees must maintain professional conduct at all times.",
     "Customer information must remain confidential.",
     "All employees must follow company safety standards.",
     "Conflicts of interest must be disclosed immediately.",
     "Employees must report unethical behavior.",
     "Company resources must be used responsibly."],
    embedding=embeddings,
)

retriever = store.as_retriever(search_kwargs={"k": 2})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

docs = retriever.invoke("What should employees do?")
context = "\n".join(d.page_content for d in docs)
answer = llm.invoke(f"Answer using only this context:\n{context}\n\nQ: What all should employees do?")
print(answer.content)