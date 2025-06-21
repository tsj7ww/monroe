from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# DATASET
class DatasetType(str, Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"

class DatasetStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

class DatasetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    dataset_type: DatasetType
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}

class DatasetCreate(DatasetBase):
    """Schema for creating a dataset."""
    pass

class DatasetUpdate(BaseModel):
    """Schema for updating a dataset."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class DatasetInDB(DatasetBase):
    """Schema for dataset in database."""
    id: str
    user_id: str
    original_filename: str
    stored_filename: str
    file_size: int
    file_type: str
    file_hash: str
    bucket_name: str
    status: DatasetStatus
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class Dataset(DatasetInDB):
    """Schema for dataset response."""
    download_url: Optional[str] = None
    
class DatasetList(BaseModel):
    """Schema for paginated dataset list."""
    total: int
    items: List[Dataset]
    page: int
    page_size: int

class DatasetUploadResponse(BaseModel):
    """Response after dataset upload."""
    id: str
    message: str = "Dataset uploaded successfully"
    filename: str
    size: int
    processing_time: float

# SQL
class SQLQueryRequest(BaseModel):
    query: str = Field(..., description="SQL query to store")
    query_name: str = Field(..., description="Name for the query")
    description: Optional[str] = None
    database_config_id: str = Field(..., description="ID of the database configuration")
    parameters: Optional[Dict[str, Any]] = Field(default={})
    tags: Optional[List[str]] = []

class SQLQueryResponse(BaseModel):
    query_id: str
    message: str
    query_name: str
    created_at: datetime

# FILE
class FileUploadResponse(BaseModel):
    file_id: str
    message: str
    filename: str
    size: int
    minio_object: str
    metadata_id: str

# HEALTH
class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime
    services: Dict[str, str]
    environment: Dict[str, Any]