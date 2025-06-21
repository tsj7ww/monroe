from minio import Minio
from minio.error import S3Error
from typing import Tuple, Optional
import hashlib
import io
import json
from datetime import timedelta
from fastapi import UploadFile, HTTPException
from app.core.config import get_settings
from app.core.dependencies import minio_client

class StorageService:
    """MinIO storage service for Monroe platform"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = minio_client
        self.bucket_name = self.settings.minio_bucket_name
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist with proper policy"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                
                # Set bucket policy if specified
                if self.settings.MINIO_BUCKET_POLICY == "public-read":
                    policy = {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"AWS": "*"},
                                "Action": ["s3:GetObject"],
                                "Resource": [f"arn:aws:s3:::{self.bucket_name}/*"]
                            }
                        ]
                    }
                    self.client.set_bucket_policy(self.bucket_name, json.dumps(policy))
                    
        except S3Error as e:
            print(f"Error creating bucket: {e}")
    
    async def upload_file(
        self, 
        file: UploadFile, 
        object_name: str,
        metadata: Optional[dict] = None
    ) -> Tuple[str, str, int]:
        """Upload file to MinIO bucket"""
        try:
            content = await file.read()
            file_size = len(content)
            file_hash = hashlib.sha256(content).hexdigest()
            
            # Add Monroe metadata
            if metadata is None:
                metadata = {}
            metadata.update({
                "project": self.settings.PROJECT_NAME,
                "environment": self.settings.ENVIRONMENT,
                "file_hash": file_hash
            })
            
            self.client.put_object(
                self.bucket_name,
                object_name,
                io.BytesIO(content),
                file_size,
                content_type=file.content_type or "application/octet-stream",
                metadata=metadata
            )
            
            return object_name, file_hash, file_size
            
        except S3Error as e:
            raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")
    
    def get_download_url(self, object_name: str, expires: int = 3600) -> str:
        """Generate presigned URL for downloading"""
        try:
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(seconds=expires)
            )
            return url
        except S3Error as e:
            raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    
    def get_upload_url(self, object_name: str, expires: int = 3600) -> str:
        """Generate presigned URL for direct upload"""
        try:
            url = self.client.presigned_put_object(
                self.bucket_name,
                object_name,
                expires=timedelta(seconds=expires)
            )
            return url
        except S3Error as e:
            raise HTTPException(status_code=500, detail=f"Could not generate upload URL: {str(e)}")
    
    def delete_file(self, object_name: str) -> bool:
        """Delete file from storage"""
        try:
            self.client.remove_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    def list_files(self, prefix: str = "", limit: int = 100) -> list:
        """List files in bucket with optional prefix"""
        try:
            objects = self.client.list_objects(
                self.bucket_name,
                prefix=prefix,
                recursive=True
            )
            files = []
            for obj in objects:
                if len(files) >= limit:
                    break
                files.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified
                })
            return files
        except S3Error as e:
            raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

storage_service = StorageService()