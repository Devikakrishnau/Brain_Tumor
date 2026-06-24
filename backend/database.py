import os
from pymongo import MongoClient

# Fallback URI if not set
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client.agentic_tumor_db
    patients_collection = db.patients
    cases_collection = db.cases
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"⚠️ Could not connect to MongoDB: {e}")
    client = None
    db = None

def save_patient_record(record: dict):
    if cases_collection is not None:
        return cases_collection.insert_one(record).inserted_id
    print("⚠️ MongoDB not connected. Record not saved.")
    return None

def get_similar_cases(diagnosis: str, limit: int = 3):
    if cases_collection is not None:
        # In a real scenario, this would use pgvector or MongoDB vector search 
        # on embeddings. For now, doing a basic text match.
        return list(cases_collection.find({"probable_diagnosis": {"$regex": diagnosis, "$options": "i"}}).limit(limit))
    return []
