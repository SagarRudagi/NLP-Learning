"""
Contextual Compression Retriever (CCR) is a retrieval strategy where you first
fetch candidate documents, then compress or filter them using the query context
before sending them to the LLM. Instead of passing full raw documents, CCR
keeps only the most relevant sentences/chunks, which reduces token usage and
often improves answer focus.

How it helps:
1) Better signal-to-noise ratio: irrelevant parts of retrieved docs are removed.
2) Lower cost and latency: fewer tokens are sent to the generation model.
3) Improved answer quality: the model sees cleaner, query-specific evidence.
4) Works well with long documents where only small sections are useful.

Common problems / trade-offs:
1) Information loss risk: an aggressive compressor may drop important context.
2) Extra pipeline complexity: retrieval + compression adds more moving parts.
3) Additional runtime overhead: compression itself takes time and may use an LLM.
4) Error propagation: if initial retrieval is weak, compression cannot recover
   missing facts and can amplify the retrieval mistake.
5) Tuning sensitivity: chunk size, compression prompt/rules, and thresholds
   need careful tuning for each domain.

Practical guideline:
Use contextual compression when retrieved documents are verbose or noisy, but
validate on real queries to ensure the compressor is not removing key evidence.

About LLMChainExtractor:
LLMChainExtractor is a common compressor used with CCR. It receives the user
query and retrieved documents, then extracts only the query-relevant lines from
each document. This improves focus and reduces tokens, but if configured too
aggressively it may remove useful context.
"""

import os
# Workaround for OpenMP duplicate runtime initialization seen with FAISS on some macOS setups.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables such as API keys from .env.
load_dotenv()


# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# Create a FAISS vector store from the documents
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Set up the compressor using an LLM
llm = ChatOpenAI(model="gpt-4")
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

query = "What is photosynthesis?"
results = compression_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
