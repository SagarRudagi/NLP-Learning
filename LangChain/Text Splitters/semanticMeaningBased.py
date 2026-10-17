"""Example of using SemanticChunker for meaning-based text splitting.

This splitter focuses on semantic similarity instead of a fixed character count.
It tries to detect when the meaning changes and then creates a new chunk, which
is useful for long documents where topic shifts matter more than raw length.
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables such as API keys from a .env file.
load_dotenv()

# Sample paragraph that contains two different topics: farming and terrorism.
# A semantic splitter can detect that the topic changes and split at that boundary.
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# SemanticChunker uses embeddings to measure how similar consecutive sentences or sections are.
# When the semantic similarity drops enough, it creates a break, which is better for meaning-based segmentation.
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), # Embeddings model used to compute semantic similarity between text segments
    breakpoint_threshold_type="standard_deviation", # Method to determine when a semantic break occurs based on standard deviation of similarity scores
    breakpoint_threshold_amount=1.0, # The threshold value for determining a semantic break based on the chosen method
)

# Create chunked documents based on semantic boundaries.
chunks = text_splitter.create_documents([sample])
print(len(chunks))
print(chunks)