import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore

load_dotenv()

# ----------------------------
# ENV VALIDATION
# ----------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is not set")

if not PINECONE_INDEX:
    raise ValueError("❌ PINECONE_INDEX is not set")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is not set")


# ----------------------------
# INIT PINECONE (Modern SDK)
# ----------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ----------------------------
# EMBEDDINGS (Reuse instance)
# ----------------------------

embeddings = OpenAIEmbeddings()

# ----------------------------
# VECTORSTORE (Reuse instance)
# ----------------------------

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",  # Must match your stored metadata key
)


# ----------------------------
# QUERY FUNCTION
# ----------------------------

def query_pinecone(query: str, user_id: str):
    """
    Query Pinecone using metadata filter (userId).
    Returns filtered LangChain Document objects.
    """

    if not query:
        return []

    try:
        print(f"\n🔎 Pinecone Query | userId={user_id}")
        print("Query:", query)

        # Search top 4 matches filtered by userId metadata
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=4,
            filter={"userId": user_id},
        )

        # Lower score = better match (cosine distance)
        SCORE_THRESHOLD = 1.0

        filtered_docs = [
            doc for doc, score in results if score < SCORE_THRESHOLD
        ]

        print(f"✅ Retrieved {len(filtered_docs)} relevant docs")

        return filtered_docs

    except Exception as e:
        print("❌ Pinecone error:", str(e))
        return []
