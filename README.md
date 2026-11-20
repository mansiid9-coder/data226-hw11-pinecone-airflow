# DATA 226 – HW11: Pinecone job as an Airflow job

Mansi Verma · SJSU · Fall 2026

DAG `pinecone_semantic_search`: `prepare_data → create_index → embed_and_upsert → test_search`

- Embeds 12 short texts with `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Stores them in a Pinecone serverless index (`semantic-search-hw`, cosine)
- Runs a semantic query and logs the top-3 matches

## Run
1. `docker compose down && docker compose up -d` (compose file installs `sentence-transformers` and `pinecone`)
2. Airflow UI → Admin → Variables → add `pinecone_api_key`
3. Trigger the DAG `pinecone_semantic_search`

## Package install note
`docker-compose.yaml` adds `sentence-transformers pinecone` to the `_PIP_ADDITIONAL_REQUIREMENTS` default.
My `.env` already set that variable (for the Snowflake provider), which overrides the compose default,
so the same packages were added there too (see `env.example`).
