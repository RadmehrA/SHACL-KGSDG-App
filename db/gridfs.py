import gridfs
from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")

client = MongoClient(MONGO_URI)
db = client["kg_llm_db"]

fs = gridfs.GridFS(db)
