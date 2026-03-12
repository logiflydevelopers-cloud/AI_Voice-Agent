import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

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
# INIT PINECONE
# ----------------------------

print("🚀 Initializing Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


# ----------------------------
# EMBEDDINGS (reuse instance)
# ----------------------------

print("🧠 Loading embeddings...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # faster + cheaper
)


# ----------------------------
# VECTOR STORE
# ----------------------------

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",  # must match metadata key
)

print("✅ Pinecone vectorstore ready")


# ----------------------------
# WARMUP (removes cold start)
# ----------------------------

try:
    print("🔥 Warming Pinecone connection...")
    vectorstore.similarity_search("warmup", k=1)
    print("✅ Pinecone warmup complete")
except Exception as e:
    print("⚠️ Pinecone warmup failed:", e)


# ----------------------------
# QUERY FUNCTION
# ----------------------------

def query_pinecone(query: str, user_id: str):
    """
    Query Pinecone using metadata filter (userId).
    Returns LangChain Documents.
    """

    if not query:
        return []

    try:
        print("\n🔎 Pinecone Query")
        print("User:", user_id)
        print("Query:", query)

        results = vectorstore.similarity_search_with_score(
            query=query,
            k=3,  # lower = faster for voice agents
            filter={"userId": {"$eq": user_id}},
        )

        # Lower score = better match (cosine distance)
        SCORE_THRESHOLD = 1.0

        filtered_docs = [
            doc for doc, score in results if score < SCORE_THRESHOLD
        ]

        print(f"📄 Retrieved {len(filtered_docs)} relevant docs")

        return filtered_docs

    except Exception as e:
        print("❌ Pinecone error:", str(e))
        return []