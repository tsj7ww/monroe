from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from datetime import datetime

from app.core.config import get_settings
from app.core.dependencies import mongo_client, minio_client
# from app.services.storage import storage_service
from app.schemas.data import (
    # DatabaseConfig,
    # DatabaseConfigResponse,
    SQLQueryRequest,
    SQLQueryResponse,
    # FileUploadResponse,
    HealthResponse
)
# from app.core.dependencies import (
#     get_db_configs_collection,
#     get_queries_collection,
#     get_files_collection
# )
from app.utils.data_init import initialize_data

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check for MongoDB and MinIO services"""
    settings = get_settings()
    services_status = {}
    
    # Check MinIO
    try:
        minio_client.list_buckets()
        services_status["minio"] = "healthy"
    except Exception as e:
        services_status["minio"] = f"unhealthy: {str(e)}"
    
    # Check MongoDB
    try:
        await mongo_client.admin.command('ping')
        services_status["mongodb"] = "healthy"
    except Exception as e:
        services_status["mongodb"] = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if all(s == "healthy" for s in services_status.values()) else "degraded"
    
    environment_info = {
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "environment": settings.ENVIRONMENT,
        "minio_endpoint": settings.minio_endpoint,
        "minio_bucket": settings.minio_bucket_name,
        "mongodb_database": settings.mongodb_database_name,
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
        "allowed_extensions": settings.allowed_extensions_list,
        "services": settings.get_service_urls()
    }
    
    return HealthResponse(
        status=overall_status,
        message=f"{settings.PROJECT_NAME} API is running",
        timestamp=datetime.utcnow(),
        services=services_status,
        environment=environment_info
    )


@router.get("/init")
async def initialize():
    """Initialize the application with default settings and configurations."""
    initialize_data()
    return {"message": "Application initialized with default data and configurations."}


# @router.post("/post/database", response_model=DatabaseConfigResponse)
# async def create_database_config(
#     config: DatabaseConfig,
#     collection = Depends(get_db_configs_collection)
# ):
#     """Store database configuration in MongoDB."""
#     settings = get_settings()
    
#     # Check if config exists
#     existing = await collection.find_one({"name": config.name})
#     if existing:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Configuration '{config.name}' already exists"
#         )
    
#     # Hash password
#     password_hash = hashlib.sha256(
#         f"{config.password}{settings.SECRET_KEY}".encode()
#     ).hexdigest()
    
#     config_doc = {
#         "name": config.name,
#         "db_type": config.db_type.value,
#         "host": config.host,
#         "port": config.port,
#         "database": config.database,
#         "username": config.username,
#         "password_hash": password_hash,
#         "options": config.options,
#         "description": config.description,
#         "active": config.active,
#         "created_at": datetime.utcnow(),
#         "updated_at": datetime.utcnow()
#     }
    
#     result = await collection.insert_one(config_doc)
    
#     return DatabaseConfigResponse(
#         config_id=str(result.inserted_id),
#         message="Database configuration created successfully",
#         name=config.name,
#         db_type=config.db_type.value,
#         created_at=config_doc["created_at"]
#     )

# @router.post("/post/sql", response_model=SQLQueryResponse)
# async def store_sql_query(
#     query_request: SQLQueryRequest,
#     queries_collection = Depends(get_queries_collection),
#     configs_collection = Depends(get_db_configs_collection)
# ):
#     """Store SQL query in MongoDB."""
#     # Validate database config exists
#     db_config = await configs_collection.find_one({"_id": query_request.database_config_id})
#     if not db_config:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Database configuration not found"
#         )
    
#     # Basic SQL validation
#     query_lower = query_request.query.strip().lower()
#     if not query_lower:
#         raise HTTPException(status_code=400, detail="Query cannot be empty")
    
#     valid_starts = ['select', 'insert', 'update', 'delete', 'create', 'alter', 'drop', 'with']
#     if not any(query_lower.startswith(start) for start in valid_starts):
#         raise HTTPException(status_code=400, detail="Invalid SQL query")
    
#     # Check for duplicates
#     existing = await queries_collection.find_one({"query_name": query_request.query_name})
#     if existing:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Query '{query_request.query_name}' already exists"
#         )
    
#     query_doc = {
#         "query_name": query_request.query_name,
#         "query": query_request.query,
#         "description": query_request.description,
#         "database_config_id": query_request.database_config_id,
#         "database_config_name": db_config["name"],
#         "parameters": query_request.parameters,
#         "tags": query_request.tags,
#         "created_at": datetime.utcnow(),
#         "updated_at": datetime.utcnow()
#     }
    
#     result = await queries_collection.insert_one(query_doc)
    
#     return SQLQueryResponse(
#         query_id=str(result.inserted_id),
#         message="SQL query stored successfully",
#         query_name=query_request.query_name,
#         created_at=query_doc["created_at"]
#     )


# @router.post("/post/file", response_model=FileUploadResponse)
# async def upload_file(
#     file: UploadFile = File(...),
#     name: str = Form(...),
#     description: Optional[str] = Form(None),
#     category: Optional[str] = Form(None),
#     tags: Optional[str] = Form(None),
#     collection = Depends(get_files_collection)
# ):
#     """Upload a file to Monroe's MinIO storage"""
#     settings = get_settings()
    
#     # Validate file extension
#     file_ext = os.path.splitext(file.filename)[1].lower()
#     if file_ext not in settings.allowed_extensions_list:
#         raise HTTPException(
#             status_code=400,
#             detail=f"File type {file_ext} not allowed. Allowed types: {settings.allowed_extensions_list}"
#         )
    
#     # Check file size
#     file.file.seek(0, 2)
#     file_size = file.file.tell()
#     file.file.seek(0)
    
#     if file_size > settings.max_upload_size_bytes:
#         raise HTTPException(
#             status_code=400,
#             detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB"
#         )
    
#     # Generate unique filename with Monroe prefix
#     timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#     object_name = f"{settings.ENVIRONMENT}/{timestamp}_{file.filename}"
    
#     # Upload to MinIO with metadata
#     _, file_hash, actual_size = await storage_service.upload_file(
#         file, 
#         object_name,
#         metadata={
#             "name": name,
#             "category": category or "uncategorized",
#             "uploaded_by": "api"  # In production, get from auth
#         }
#     )
    
#     # Store metadata in MongoDB
#     metadata = {
#         "name": name,
#         "original_filename": file.filename,
#         "object_name": object_name,
#         "size": actual_size,
#         "content_type": file.content_type,
#         "file_hash": file_hash,
#         "description": description,
#         "category": category,
#         "tags": tags.split(",") if tags else [],
#         "bucket": settings.minio_bucket_name,
#         "project": settings.PROJECT_NAME,
#         "environment": settings.ENVIRONMENT,
#         "created_at": datetime.utcnow(),
#         "updated_at": datetime.utcnow()
#     }
    
#     result = await collection.insert_one(metadata)
    
#     return FileUploadResponse(
#         file_id=str(result.inserted_id),
#         message=f"File uploaded successfully to {settings.PROJECT_NAME}",
#         filename=file.filename,
#         size=actual_size,
#         minio_object=object_name,
#         metadata_id=str(result.inserted_id)
#     )