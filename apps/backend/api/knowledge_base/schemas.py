from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class FilePayload(BaseModel):
    filename: str
    content: str
    size: int = 0

class UploadRequest(BaseModel):
    files: List[FilePayload] = Field(..., min_length=1)
    
class FileMetaOut(BaseModel):
    filename: str
    size: int
    row_count: int

class UploadResponse(BaseModel):
    upload_id: str
    document_count: int
    file_metadata: List[FileMetaOut]
    preview: List[Dict[str, Any]] = []

class ValidationError(BaseModel):
    type: str
    message: str
    fields: Optional[List[str]] = None
    count: Optional[str] = None

class ValidationReport(BaseModel):
    is_valid: bool
    total_records: int
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []
    source_fields: List[str] = []
    detected_mapping: Dict[str, str] = {}
    target_fields: Dict[str, str] = {}

class FieldMappingRequest(BaseModel):
    mapping: Dict[str, str]

class DatasetOut(BaseModel):
    id: str
    version_number: int
    collection_name: str
    document_count: int
    segment_count: int
    is_active: bool
    status: str
    created_at: str

class DatasetListResponse(BaseModel):
    datasets: List[DatasetOut]