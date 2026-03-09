import os
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")

client = AsyncIOMotorClient(MONGO_URI)
llm_db = client["kg_llm_db"]
db = client["synthetic_models"]
ontology_collection = llm_db["ontologies"]
model_collection = llm_db["models"]
gan_collection = db["gan_models"]
vae_collection = db["vae_models"]
db_ont = client["ontologies"]
ontology_collection = db_ont["parsed_ontologies"]
tensor_collection = db_ont["graph_tensors"]
embedding_collection = db_ont["hybrid_embeddings"]
shacl_collection = db_ont["shacl_constraints"]
distribution_collection = db_ont["distributional_embeddings"]

parsed_shacl_col = db["parsed_shacl"]
parsed_ontology_col = db["parsed_ontology"]