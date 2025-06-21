# database init code - mongodb and minio
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from minio import Minio
from minio.error import S3Error

def init_mongodb():
    """Initialize MongoDB connection."""
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    try:
        client = MongoClient(mongo_uri)
        client.admin.command('ping')  # Test the connection
        print("MongoDB connection established successfully.")
        return client
    except ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise
def init_minio():
    """Initialize MinIO connection."""
    minio_url = os.getenv('MINIO_URL', 'localhost:9000')
    access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
    try:
        client = Minio(
            minio_url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False  # Set to True if using HTTPS
        )
        # add bucket
        if not client.bucket_exists('monroe-data'):
            client.make_bucket('monroe-data')
        return client
    except S3Error as e:
        print(f"Failed to connect to MinIO: {e}")
        raise
def model_config_data():
    """Return configuration data for the model."""
    return [
        {"model_id": "monroe_model_001",
         "model_name": "monroe",
         "version": "1.0",
         "description": "Monroe backend database initialization script."},
        {"model_id": "monroe_model_002",
         "model_name": "monroe",
         "version": "1.1",
         "description": "Updated configuration for Monroe backend."},
        {"model_id": "monroe_model_003",
         "model_name": "monroe",
         "version": "1.2",
         "description": "Latest configuration for Monroe backend."}
        ]
def get_data_files():
    """Return a list of data files to be uploaded to MinIO."""
    return [
        "/data/creditcard.csv",
    ]
def init_db():
    """Initialize both MongoDB and MinIO."""
    try:
        mongo_client = init_mongodb()
        model_configs = model_config_data()
        # insert model configuration data into MongoDB
        db = mongo_client['monroe']
        config_collection = db['model_configs']
        config_collection.insert_many(model_configs)

        minio_client = init_minio()
        data_files = get_data_files()
        # Upload data files to MinIO
        bucket_name = 'monroe-data'
        for file_path in data_files:
            # Upload the file
            minio_client.fput_object(bucket_name, os.path.basename(file_path), file_path)
            print(f"Uploaded {file_path} to MinIO bucket {bucket_name}.")

    except Exception as e:
        print(f"Database initialization failed: {e}")
        raise
if __name__ == "__main__":
    # Run the database initialization
    try:
        init_db()
        print("Database initialization completed successfully.")
    except Exception as e:
        print(f"Error during database initialization: {e}")
        exit(1)