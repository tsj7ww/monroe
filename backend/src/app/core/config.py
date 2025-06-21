import os
from typing import List
from functools import lru_cache

class Settings:
    """
    Monroe platform settings from environment variables.
    Only uses MinIO and MongoDB as databases.
    """
    def __init__(self):
        # Project Information
        self.PROJECT_NAME = os.environ["PROJECT_NAME"]
        self.PROJECT_DESCRIPTION = os.environ["PROJECT_DESCRIPTION"]
        self.PROJECT_VERSION = os.environ["PROJECT_VERSION"]
        self.ENVIRONMENT = os.environ["ENVIRONMENT"]
        
        # API Configuration
        self.API_PORT = int(os.environ["API_PORT"])
        self.API_HOST = os.environ["API_HOST"]
        self.API_URL = os.environ["API_URL"]
        
        # Frontend Configuration
        self.FRONTEND_PORT = int(os.environ["FRONTEND_PORT"])
        self.FRONTEND_HOST = os.environ["FRONTEND_HOST"]
        self.FRONTEND_URL = os.environ["FRONTEND_URL"]
        
        # MongoDB Configuration
        self.MONGODB_HOST = os.environ["MONGODB_HOST"]
        self.MONGODB_PORT = int(os.environ["MONGODB_PORT"])
        self.MONGODB_USER = os.environ["MONGODB_USER"]
        self.MONGODB_PASSWORD = os.environ["MONGODB_PASSWORD"]
        self.MONGODB_DATABASE = os.environ["MONGODB_DATABASE"]
        self.MONGODB_CONNECTION_STRING = os.environ["MONGODB_CONNECTION_STRING"]
        self.MONGODB_PYMONGO_URI = os.environ["MONGODB_PYMONGO_URI"]
        
        # MinIO Configuration
        self.MINIO_ROOT_USER = os.environ["MINIO_ROOT_USER"]
        self.MINIO_ROOT_PASSWORD = os.environ["MINIO_ROOT_PASSWORD"]
        self.MINIO_HOST = os.environ["MINIO_HOST"]
        self.MINIO_PORT = int(os.environ["MINIO_PORT"])
        self.MINIO_CONSOLE_PORT = int(os.environ["MINIO_CONSOLE_PORT"])
        self.MINIO_SECURE = os.environ["MINIO_SECURE"].lower() == "true"
        self.MINIO_CONNECTION_STRING = os.environ["MINIO_CONNECTION_STRING"]
        self.MINIO_BUCKET = os.environ["MINIO_BUCKET"]
        self.MINIO_BUCKET_POLICY = os.environ["MINIO_BUCKET_POLICY"]
        self.MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
        self.MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
        
        # File Upload Settings
        self.MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "100"))
        self.ALLOWED_FILE_EXTENSIONS = os.environ.get(
            "ALLOWED_FILE_EXTENSIONS", 
            ".csv,.json,.xlsx,.parquet,.txt,.pdf,.pkl,.h5,.hdf5,.npy,.npz"
        )
        
        # Security
        self.SECRET_KEY = os.environ.get(
            "SECRET_KEY", 
            "monroe-secret-key-change-in-production"
        )
        self.ALGORITHM = os.environ.get("ALGORITHM", "HS256")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(
            os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
        )
    
    # Computed properties
    @property
    def minio_endpoint(self) -> str:
        """MinIO endpoint for SDK connections"""
        return f"{self.MINIO_HOST}:{self.MINIO_PORT}"
    
    @property
    def max_upload_size_bytes(self) -> int:
        """Convert MB to bytes"""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Convert comma-separated extensions to list"""
        return [ext.strip() for ext in self.ALLOWED_FILE_EXTENSIONS.split(",")]
    
    @property
    def cors_origins(self) -> List[str]:
        """CORS allowed origins"""
        origins = [
            self.FRONTEND_URL,
            f"{self.FRONTEND_HOST}:{self.FRONTEND_PORT}",
            self.API_URL,
            f"{self.API_HOST}:{self.API_PORT}",
            "http://localhost:3000",  # Development frontend
            "http://localhost:8000",  # Development API
        ]
        return list(set(origins))  # Remove duplicates
    
    @property
    def mongodb_database_name(self) -> str:
        """MongoDB database name for consistency"""
        return self.MONGODB_DATABASE
    
    @property
    def minio_bucket_name(self) -> str:
        """MinIO bucket name for consistency"""
        return self.MINIO_BUCKET
    
    def get_service_urls(self) -> dict:
        """Get all service URLs for documentation"""
        return {
            "api": self.API_URL,
            "frontend": self.FRONTEND_URL,
            "minio_console": f"{self.API_HOST}:{self.MINIO_CONSOLE_PORT}",
        }

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()