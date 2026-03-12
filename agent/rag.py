import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

load_dotenv()

# ----------------------------
# ENV VARIABLES
# ----------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY")

if not PINECONE_INDEX:
    raise ValueError("❌ Missing PINECONE_INDEX")

print("🚀 Initializing Pinecone...")

# ----------------------------
# INIT PINECONE (NEW SDK)
# ----------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX)

print("🧠 Loading OpenAI embeddings...")

# ----------------------------
# EMBEDDINGS
# ----------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# ----------------------------
# VECTOR STORE
# ----------------------------

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

print("✅ Pinecone vectorstore ready")

# ----------------------------
# WARMUP (avoid first query delay)
# ----------------------------

try:
    print("🔥 Warming Pinecone...")
    vectorstore.similarity_search("warmup", k=1)
    print("✅ Pinecone warmed up")
except Exception as e:
    print("⚠️ Pinecone warmup skipped:", e)

# ----------------------------
# QUERY FUNCTION
# ----------------------------

def query_pinecone(query: str, user_id: str):

    if not query:
        return []

    try:
        print(f"\n🔎 Pinecone Query | userId={user_id}")
        print("Query:", query)

        results = vectorstore.similarity_search_with_score(
            query=query,
            k=3,  # lower = faster voice responses
            filter={"userId": {"$eq": user_id}},
        )

        SCORE_THRESHOLD = 1.0

        docs = [
            doc for doc, score in results if score < SCORE_THRESHOLD
        ]

        print(f"📄 Retrieved {len(docs)} relevant docs")

        return docs

    except Exception as e:
        print("❌ Pinecone error:", str(e))
        return []