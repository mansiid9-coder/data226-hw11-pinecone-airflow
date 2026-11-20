"""
DATA 226 HW11 - Pinecone job as an Airflow job (Mansi Verma)

Pipeline:  prepare_data >> create_index >> embed_and_upsert >> test_search
- Embeddings: sentence-transformers 'all-MiniLM-L6-v2' (384-dim vectors)
- Vector DB : Pinecone serverless index (free tier: aws / us-east-1)
- API key   : Airflow Variable 'pinecone_api_key'  (never hard-coded)
"""
from datetime import datetime
import json, os

try:                                    # Airflow 3
    from airflow.sdk import dag, task, Variable
except ImportError:                     # Airflow 2
    from airflow.decorators import dag, task
    from airflow.models import Variable

INDEX_NAME = "semantic-search-hw"
MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSION = 384
DATA_PATH = "/tmp/hw11_articles.json"

ARTICLES = [
    "Apache Airflow schedules and monitors data pipelines written as Python DAGs.",
    "Snowflake is a cloud data warehouse that separates storage from compute.",
    "dbt lets analysts transform data in the warehouse using SQL SELECT statements.",
    "Apache Spark processes large datasets in parallel across a cluster of machines.",
    "Adaptive Query Execution in Spark re-optimizes query plans at runtime.",
    "Kafka is a distributed log used to stream events between producers and consumers.",
    "Spark Structured Streaming treats a live data stream as an unbounded table.",
    "MLflow tracks machine learning experiments, parameters, metrics and models.",
    "A vector database stores embeddings and finds the most similar items quickly.",
    "Sentence transformers turn a sentence into a fixed-length numeric vector.",
    "NYC taxi trip data includes pickup time, location, distance and fare amount.",
    "Retrieval augmented generation feeds search results from a vector DB to an LLM.",
]


@dag(
    dag_id="pinecone_semantic_search",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    tags=["data226", "pinecone", "hw11"],
)
def pinecone_semantic_search():

    @task
    def prepare_data() -> str:
        records = [{"id": str(i), "text": t} for i, t in enumerate(ARTICLES)]
        with open(DATA_PATH, "w") as f:
            json.dump(records, f)
        print(f"Saved {len(records)} records to {DATA_PATH}")
        return DATA_PATH

    @task
    def create_index() -> str:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=Variable.get("pinecone_api_key"))
        existing = [ix["name"] for ix in pc.list_indexes()]
        if INDEX_NAME in existing:           # rerun-safe: start clean
            pc.delete_index(INDEX_NAME)
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created index {INDEX_NAME}")
        return INDEX_NAME

    @task
    def embed_and_upsert(data_path: str, index_name: str) -> int:
        import time
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer

        with open(data_path) as f:
            records = json.load(f)
        model = SentenceTransformer(MODEL_NAME)
        vectors = model.encode([r["text"] for r in records]).tolist()

        pc = Pinecone(api_key=Variable.get("pinecone_api_key"))
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
        index = pc.Index(index_name)
        index.upsert(vectors=[
            {"id": r["id"], "values": v, "metadata": {"text": r["text"]}}
            for r, v in zip(records, vectors)
        ])
        print(f"Upserted {len(records)} vectors")
        return len(records)

    @task
    def test_search(index_name: str, n_upserted: int):
        import time
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer

        query = "How do I process streaming events in real time?"
        q_vec = SentenceTransformer(MODEL_NAME).encode(query).tolist()
        index = Pinecone(api_key=Variable.get("pinecone_api_key")).Index(index_name)

        for _ in range(15):                  # wait until vectors are searchable
            if index.describe_index_stats().get("total_vector_count", 0) >= n_upserted:
                break
            time.sleep(2)

        res = index.query(vector=q_vec, top_k=3, include_metadata=True)
        print(f"Query: {query}")
        for m in res["matches"]:
            print(f"  score={m['score']:.4f}  {m['metadata']['text']}")

    path = prepare_data()
    idx = create_index()
    n = embed_and_upsert(path, idx)
    test_search(idx, n)


pinecone_semantic_search()
