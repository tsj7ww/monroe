from motor.motor_asyncio import AsyncIOMotorClient
from minio import Minio
from app.core.config import get_settings

settings = get_settings()

# MongoDB client using the connection string
mongo_client = AsyncIOMotorClient(settings.MONGODB_PYMONGO_URI)
mongodb = mongo_client[settings.mongodb_database_name]

# Collections
files_collection = mongodb.files
db_configs_collection = mongodb.database_configs
queries_collection = mongodb.sql_queries
datasets_collection = mongodb.datasets
models_collection = mongodb.ml_models
experiments_collection = mongodb.experiments

# MinIO client
minio_client = Minio(
    settings.minio_endpoint,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE
)

# Dependency functions
def get_mongodb():
    return mongodb

def get_mongo_client():
    return mongo_client

def get_minio_client():
    return minio_client

def get_files_collection():
    return files_collection

def get_db_configs_collection():
    return db_configs_collection

def get_queries_collection():
    return queries_collection

def get_datasets_collection():
    return datasets_collection

def get_models_collection():
    return models_collection

def get_experiments_collection():
    return experiments_collection