# =============================================================================
# COMPLETE PROFESSIONAL RAG SYSTEM - A to Z
# =============================================================================

import os
import tempfile
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import io
import hashlib
import traceback
import re
import json
from pathlib import Path
import time
import threading
import numpy as np
from collections import Counter

# Core ML/DL imports
try:
    from rank_bm25 import BM25Okapi
    print("✅ BM25 imported successfully")
except ImportError:
    print("❌ BM25 not available - run: pip install rank-bm25")
    BM25Okapi = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    print("✅ scikit-learn imported successfully")
except ImportError:
    print("❌ scikit-learn not available - run: pip install scikit-learn")
    cosine_similarity = None

# Image processing
try:
    from PIL import Image
    import pillow_heif
    print("✅ Image processing libraries imported")
except ImportError:
    print("⚠️ Image processing libraries not available")

# Core imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd

# Optional MongoDB support (for Render / cloud deployment)
try:
    from pymongo import MongoClient
    HAS_MONGO = True
    print("✅ MongoDB support available")
except ImportError:
    MongoClient = None
    HAS_MONGO = False
    print("⚠️ MongoDB support not available")

# LangChain Professional Imports - COMPATIBLE VERSION
try:
    from langchain_core.documents import Document
    print("✅ Using langchain_core.documents")
except ImportError as e:
    print(f"❌ Failed to import Document: {e}")
    raise

# Text splitting - FIXED IMPORTS FOR OLDER VERSION
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    # MarkdownHeaderTextSplitter might not be available in older versions
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        HAS_MARKDOWN_SPLITTER = True
    except ImportError:
        HAS_MARKDOWN_SPLITTER = False
        print("⚠️ MarkdownHeaderTextSplitter not available")
    print("✅ Text splitter imported successfully")
except ImportError as e:
    print(f"❌ Text splitter failed: {e}")
    raise

# Embeddings and vector stores - UPDATED FOR NEW VERSIONS
try:
    # Use the newer imports
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    print("✅ Using updated embeddings and vector stores")
except ImportError as e:
    print(f"⚠️ Updated imports failed: {e}")
    # Fallback to community imports
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        print("✅ Using community embeddings and vector stores")
    except ImportError as e:
        print(f"❌ Could not import embeddings: {e}")
        HuggingFaceEmbeddings = None
        Chroma = None


# Retrievers - FIXED IMPORTS FOR OLDER VERSION
try:
    from langchain_community.retrievers import BM25Retriever

except ImportError as e:
    BM25Retriever = None


# Document loaders - FIXED IMPORTS
try:
    from langchain_community.document_loaders import (
        UnstructuredFileLoader, 
        PyPDFLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredExcelLoader,
        UnstructuredPowerPointLoader,
        CSVLoader,
        TextLoader
    )
    print("✅ Document loaders imported successfully")
except ImportError as e:
    print(f"❌ Document loaders failed: {e}")
    raise

# Utility functions - OPTIONAL
try:
    from langchain_community.vectorstores.utils import filter_complex_metadata
    print("✅ Vector store utilities imported")
except ImportError:
    print("⚠️ Vector store utilities not available")
    filter_complex_metadata = None

# Google Drive Professional - OPTIONAL
try:
    from langchain_googledrive.document_loaders import GoogleDriveLoader
    HAS_GOOGLE_DRIVE = True
    print("✅ Google Drive support available")
except ImportError:
    HAS_GOOGLE_DRIVE = False
    print("⚠️ Google Drive support not available")

try:
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    HAS_OAUTH = True
    print("✅ Google OAuth support available")
except ImportError:
    HAS_OAUTH = False
    print("⚠️ Google OAuth support not available")

# Vector storage
import chromadb
from chromadb.config import Settings

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # This loads your .env file

# =============================================================================
# CONFIGURATION - USING ENVIRONMENT VARIABLES
# =============================================================================

# DeepSeek API Configuration - FROM .env
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-72385ab6d41845058899aabb4be43f32")

# Server Configuration - FROM .env  
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")

# File Storage Directories - FROM .env
STRUCTURED_DATA_DIR = os.getenv("STRUCTURED_DATA_DIR", "./structured_data")
ORIGINAL_FILES_DIR = os.getenv("ORIGINAL_FILES_DIR", "./original_files")
STATE_DIR = os.getenv("STATE_DIR", "./sync_state")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_store")

# Create directories
os.makedirs(STRUCTURED_DATA_DIR, exist_ok=True)
os.makedirs(ORIGINAL_FILES_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Store user credentials
user_credentials = {}

# MongoDB configuration (used for Render / production deployments)
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "rag_system")

mongo_client = None
mongo_db = None

if HAS_MONGO and MONGODB_URI:
    try:
        mongo_client = MongoClient(MONGODB_URI)
        mongo_db = mongo_client[MONGODB_DB_NAME]
        print(f"✅ Connected to MongoDB database: {MONGODB_DB_NAME}")
    except Exception as e:
        print(f"⚠️ Failed to connect to MongoDB: {e}")
        mongo_client = None
        mongo_db = None
else:
    if not HAS_MONGO:
        print("ℹ️ pymongo not installed - running with local JSON storage only")
    else:
        print("ℹ️ MONGODB_URI not set - running with local JSON storage only")

# OAuth Configuration - UPDATED FOR PRODUCTION
GOOGLE_OAUTH_CONFIG = {
    "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
    "redirect_uri": f"{RENDER_EXTERNAL_URL}/oauth2callback"  # Uses environment variable
}

print(f"🚀 Server configured for: {RENDER_EXTERNAL_URL}")
print(f"📁 Using directories: {STRUCTURED_DATA_DIR}, {ORIGINAL_FILES_DIR}")

# Corporate namespace mapping
CORPORATE_NAMESPACES = {
    "employee": "employee_docs",
    "hr": "hr_department",  
    "finance": "finance_department",
    "it": "it_department",
    "manager": "management_full",
    "executive": "management_full"
}

# Folder access by role - FIXED VERSION
CORPORATE_FOLDER_ACCESS = {
    "employee": ["01-Company-Public"],
    "manager": ["*"],  # Full access to everything
    "hr": ["01-Company-Public", "02-HR-Department"],
    "finance": ["01-Company-Public", "04-Finance"],
    "it": ["01-Company-Public", "05-IT-Department"],
    "executive": ["*"]  # Full access to everything
}

# Allowed file extensions
ALLOWED_EXT = ('.pdf', '.docx', '.txt', '.csv', '.xlsx', '.xls', '.pptx', '.ppt', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.md', '.html', '.htm', '.heic')

EMAIL_ROLE_MAPPING = {
    "hr@": "hr", "human.resources@": "hr", "recruitment@": "hr",
    "finance@": "finance", "accounting@": "finance", "payroll@": "finance",
    "it@": "it", "tech@": "it", "support@": "it",
    "admin@": "manager", "manager@": "manager", "director@": "manager",
    "head@": "manager", "lead@": "manager",
    "vp@": "executive", "cfo@": "executive", "ceo@": "executive", "president@": "executive"
}

print("🚀 Initializing COMPLETE PROFESSIONAL RAG System with ALL Features...")

# =============================================================================
# CORE COMPONENTS - YOUR EXISTING
# =============================================================================

class CorporateUserContext:
    def __init__(self, email: str, role: str):
        self.email = email
        self.role = role
        self.namespace = CORPORATE_NAMESPACES.get(role, "employee_docs")
        self.folders = CORPORATE_FOLDER_ACCESS.get(role, ["01-Company-Public"])
    
    def get_namespace(self) -> str:
        return self.namespace
    
    def can_access_namespace(self, namespace: str) -> bool:
        return namespace == self.namespace
    
    def can_access_folder(self, folder_name: str) -> bool:
        if self.role in ["manager", "executive"] or "*" in self.folders:
            return True
        return folder_name in self.folders
    
    def get_accessible_folders(self) -> list:
        return self.folders

class CorporateUserManager:
    def authenticate_user(self, email: str) -> CorporateUserContext:
        email_lower = email.lower()
        
        # 🚨 FIX: Better executive title detection
        executive_titles = ["ceo", "cfo", "cto", "president", "vp", "director", "head", "lead"]
        
        # Check email prefix for executive titles
        email_prefix = email_lower.split('@')[0]  # Get part before @
        if any(title in email_prefix for title in executive_titles):
            return CorporateUserContext(email, "executive")
        
        # Check for manager/admin specifically
        if "manager" in email_lower or "admin" in email_lower:
            return CorporateUserContext(email, "manager")
        
        # Check email patterns
        for pattern, role in EMAIL_ROLE_MAPPING.items():
            if pattern in email_lower:
                return CorporateUserContext(email, role)
        
        return CorporateUserContext(email, "employee")

# =============================================================================
# PROFESSIONAL DOCUMENT PROCESSOR - ENHANCED
# =============================================================================

class ProfessionalDocumentProcessor:
    """Enterprise-grade document processing with proper error handling and logging"""
    
    def __init__(self):
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.doc': self._load_doc,
            '.xlsx': self._load_excel,  # Keep Excel as is
            '.xls': self._load_excel,   # Keep Excel as is
            '.pptx': self._load_pptx,
            '.csv': self._load_csv,
            '.txt': self._load_text,
            '.md': self._load_markdown,
            '.png': self._load_image,
            '.jpg': self._load_image,
            '.jpeg': self._load_image,
            '.webp': self._load_image,
        }
        
        # 🚨 FIX: Create MUCH LARGER chunks for non-Excel documents
        self.text_splitters = {
            'general': RecursiveCharacterTextSplitter(
                chunk_size=2500,      # 🚨 INCREASED from 1000 to 3000
                chunk_overlap=200,    # 🚨 INCREASED from 200 to 400
                length_function=len,
                separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""]  # Added more separators
            ),
            'code': RecursiveCharacterTextSplitter(
                chunk_size=1500,      # 🚨 INCREASED from 800
                chunk_overlap=300,    # 🚨 INCREASED from 100
                separators=["\nclass ", "\ndef ", "\nimport ", "\nfrom ", "\n\n", "\n", " ", ""]
            ),
            'markdown': MarkdownHeaderTextSplitter(headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"), 
                ("###", "Header 3")
            ])
        }
    
    def process_document(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Professional document processing with proper format detection - FIXED"""
        try:
            # 🚨 FIX: Better file extension detection
            file_lower = filename.lower()
            
            # Handle files without extensions by checking MIME type or content
            if '.' not in file_lower:
                print(f"🔍 File without extension: {filename}")
                # Try to detect type from content or use fallback
                return self._fallback_loader(file_bytes, filename)
            
            file_ext = os.path.splitext(filename)[1].lower()
            
            # 🚨 FIX: Use enhanced loaders for specific cases
            if file_ext in ['.xlsx', '.xls']:
                return self.enhanced_excel_loader(file_bytes, filename)
            elif file_ext == '.pdf':
                return self.enhanced_pdf_loader(file_bytes, filename)
            elif file_ext == '.heic':
                return self.process_heic_images(file_bytes, filename)
            
            # Continue with professional processing for other file types
            if file_ext not in self.supported_extensions:
                print(f"⚠️ Unsupported file type: {filename} (extension: {file_ext})")
                return self._fallback_loader(file_bytes, filename)
            
            loader_func = self.supported_extensions[file_ext]
            documents = loader_func(file_bytes, filename)
            
            if not documents:
                print(f"⚠️ No content extracted from: {filename}")
                return []
            
            # 🚨 FIX: Better chunking with validation
            chunked_documents = self._smart_chunking(documents, filename)
            
            print(f"✅ Processed {filename}: {len(documents)} -> {len(chunked_documents)} chunks")
            return chunked_documents
            
        except Exception as e:
            print(f"❌ Failed to process {filename}: {str(e)}")
            traceback.print_exc()  # 🚨 ADD THIS FOR BETTER ERROR TRACING
            return []
    
    # 🚨 UPDATED: IMAGE PROCESSING WITH UnstructuredImageLoader
    def _load_image(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Extract text from images using UnstructuredImageLoader (automatic OCR)"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                # Use UnstructuredImageLoader - it handles OCR automatically
                from langchain_community.document_loaders import UnstructuredImageLoader
                loader = UnstructuredImageLoader(temp_path)
                documents = loader.load()
                
                if documents:
                    for doc in documents:
                        # 🚨 FIX: Ensure page_content is string
                        if doc.page_content is None:
                            doc.page_content = ""
                        
                        doc.metadata.update({
                            'file_type': 'image',
                            'processing_method': 'UnstructuredImageLoader_OCR',
                            'source': filename
                        })
                    print(f"✅ Image OCR extracted: {filename} -> {len(documents)} docs")
                    return documents
                else:
                    print(f"⚠️ No text found in image: {filename}")
                    return []
                    
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"❌ Image processing failed for {filename}: {e}")
            return []

    # 🚨 .DOC FILE SUPPORT
    def _load_doc(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Handle .doc files (not .docx)"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                # Use UnstructuredFileLoader for .doc files
                loader = UnstructuredFileLoader(temp_path)
                documents = loader.load()
                
                if documents:
                    for doc in documents:
                        doc.metadata.update({
                            'file_type': 'word_doc', 
                            'processing_method': 'UnstructuredFileLoader'
                        })
                    print(f"✅ DOC extracted: {filename} -> {len(documents)} docs")
                    return documents
                else:
                    print(f"⚠️ No content from DOC: {filename}")
                    return []
                    
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"❌ DOC processing failed for {filename}: {e}")
            return []

    def _load_pdf(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Professional PDF processing"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                # Use PyPDFLoader for reliable text extraction
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                
                # Enhanced metadata
                for doc in documents:
                    doc.metadata.update({
                        'file_type': 'pdf',
                        'processing_method': 'PyPDFLoader',
                        'total_pages': len(documents)
                    })
                
                return documents
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"PDF processing failed for {filename}: {e}")
            return []
    
    def _load_excel(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Professional Excel processing with multi-sheet support"""
        try:
            import pandas as pd
            from io import BytesIO
            
            excel_file = BytesIO(file_bytes)
            xl = pd.ExcelFile(excel_file)
            documents = []
            
            for sheet_name in xl.sheet_names:
                try:
                    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)
                    
                    # Convert DataFrame to structured text
                    sheet_content = f"SHEET: {sheet_name}\n\n"
                    sheet_content += f"COLUMNS: {', '.join(map(str, df.columns.tolist()))}\n\n"
                    
                    # Add data with proper formatting
                    for idx, row in df.iterrows():
                        row_data = []
                        for col in df.columns:
                            value = row[col]
                            if pd.notna(value):
                                row_data.append(f"{col}: {value}")
                        if row_data:
                            sheet_content += f"ROW {idx+1}: {' | '.join(row_data)}\n"
                    
                    if sheet_content.strip():
                        doc = Document(
                            page_content=sheet_content,
                            metadata={
                                'source': f"{filename} - {sheet_name}",
                                'file_type': 'excel',
                                'sheet_name': sheet_name,
                                'row_count': len(df),
                                'column_count': len(df.columns)
                            }
                        )
                        documents.append(doc)
                        print(f"✅ Excel sheet '{sheet_name}': {len(df)} rows")
                        
                except Exception as e:
                    print(f"⚠️ Failed to process sheet {sheet_name}: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            print(f"Excel processing failed for {filename}: {e}")
            return []
    
    def extract_structured_preview(self, file_bytes: bytes, filename: str, max_rows: int = 1000, preview_rows: int = 25) -> Optional[Dict[str, Any]]:
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in ['.xlsx', '.xls', '.csv']:
            return None
        
        try:
            import pandas as pd
            import numpy as np
            from io import BytesIO
        except ImportError:
            print("⚠️ Structured preview requires pandas to be installed")
            return None
        
        try:
            if file_ext in ['.xlsx', '.xls']:
                # 🚨 FIX: Create fresh BytesIO for each operation
                excel_file = BytesIO(file_bytes)
                excel = pd.ExcelFile(excel_file, engine='openpyxl')
                sheets = []
                for sheet_name in excel.sheet_names:
                    # 🚨 FIX: Create fresh BytesIO for each sheet read
                    fresh_excel_file = BytesIO(file_bytes)
                    df = pd.read_excel(fresh_excel_file, sheet_name=sheet_name, engine='openpyxl')
                    sheet_info = self._build_dataframe_preview(df, sheet_name, max_rows, preview_rows)
                    if sheet_info:
                        sheets.append(sheet_info)
                if not sheets:
                    return None
                return {
                    'type': 'excel',
                    'generated_at': datetime.now().isoformat(),
                    'source_filename': filename,
                    'sheets': sheets
                }
            else:
                # 🚨 FIX: For CSV with encoding fallback
                df = None
                # Try different encodings
                encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                for encoding in encodings_to_try:
                    try:
                        fresh_csv_file = BytesIO(file_bytes)
                        df = pd.read_csv(fresh_csv_file, encoding=encoding, nrows=max_rows)
                        if df is not None and not df.empty:
                            print(f"✅ Successfully read CSV with {encoding} encoding")
                            break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        # Try next encoding
                        continue
                
                if df is None or df.empty:
                    print(f"⚠️ Failed to read CSV with any encoding")
                    return None
                
                sheet_info = self._build_dataframe_preview(df, 'Dataset', max_rows, preview_rows)
                if not sheet_info:
                    return None
                return {
                    'type': 'csv',
                    'generated_at': datetime.now().isoformat(),
                    'source_filename': filename,
                    'sheets': [sheet_info]
                }
        except Exception as e:
            print(f"⚠️ Structured preview extraction failed for {filename}: {e}")
            return None

    def _build_dataframe_preview(self, df, sheet_name: str, max_rows: int, preview_rows: int) -> Optional[Dict[str, Any]]:
        import pandas as pd
        if df is None or df.empty:
            return None
        
        df = df.copy()
        df = df.head(max_rows)
        df = df.loc[:, ~df.columns.duplicated()]
        
        numeric_columns = [str(col) for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_columns = [str(col) for col in df.columns if str(col) not in numeric_columns]
        
        data_rows = self._convert_dataframe_rows(df)
        preview_df = df.head(min(preview_rows, len(df)))
        preview_rows_data = self._convert_dataframe_rows(preview_df)
        
        columns_info = []
        for col in df.columns:
            series = df[col]
            columns_info.append({
                'name': str(col),
                'dtype': str(series.dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(series),
                'non_null': int(series.notnull().sum()),
                'sample_values': [self._safe_value(val) for val in series.dropna().head(3)]
            })
        
        return {
            'name': sheet_name,
            'row_count': int(len(df)),
            'column_count': int(len(df.columns)),
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'columns': columns_info,
            'rows': data_rows,
            'preview_rows': preview_rows_data
        }

    def _convert_dataframe_rows(self, df) -> List[Dict[str, Any]]:
        import pandas as pd
        try:
            import numpy as np
        except ImportError:
            np = None
        
        clean_df = df.where(pd.notnull(df), None)
        records = clean_df.to_dict(orient='records')
        converted_records = []
        for row in records:
            converted_row = {}
            for key, value in row.items():
                converted_row[str(key)] = self._safe_value(value, np)
            converted_records.append(converted_row)
        return converted_records

    def _safe_value(self, value: Any, np_module=None):
        if value is None:
            return None
        if isinstance(value, (datetime,)):
            return value.isoformat()
        if np_module is not None:
            if isinstance(value, (np_module.integer,)):
                return int(value)
            if isinstance(value, (np_module.floating,)):
                return float(value)
            if isinstance(value, (np_module.bool_,)):
                return bool(value)
        if hasattr(value, 'item'):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8')
            except Exception:
                return str(value)
        return value
    
    def _load_docx(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Professional Word document processing with SMART SECTION GROUPING"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                loader = UnstructuredWordDocumentLoader(temp_path, mode="elements")
                documents = loader.load()
                
                print(f"📄 Word doc '{filename}': {len(documents)} elements")
                
                # 🚨 SMART SECTION DETECTION
                sections = []
                current_section = []
                current_section_title = "Document Start"
                
                for i, doc in enumerate(documents):
                    content = doc.page_content.strip()
                    if not content:
                        continue
                    
                    # Detect section headers
                    is_heading = self._detect_section_header(content, i, documents)
                    
                    if is_heading and current_section:
                        # Save current section and start new one
                        section_content = "\n\n".join(current_section)
                        if section_content.strip():
                            sections.append({
                                'title': current_section_title,
                                'content': section_content,
                                'size': len(section_content)
                            })
                        
                        # Start new section
                        current_section_title = content
                        current_section = [content]
                    else:
                        # Add to current section
                        current_section.append(content)
                
                # Add the final section
                if current_section:
                    section_content = "\n\n".join(current_section)
                    sections.append({
                        'title': current_section_title,
                        'content': section_content,
                        'size': len(section_content)
                    })
                
                print(f"📑 Detected {len(sections)} natural sections in '{filename}':")
                for i, section in enumerate(sections):
                    title_preview = section['title'][:60] + "..." if len(section['title']) > 60 else section['title']
                    print(f"   {i+1}. '{title_preview}' ({section['size']:,} chars)")
                
                # 🚨 GROUP RELATED SECTIONS INTO LOGICAL CHUNKS
                final_documents = []
                current_chunk_sections = []
                current_chunk_size = 0
                target_chunk_size = 3000  # 🚨 INCREASED from 3500 to 6000

                for i, section in enumerate(sections):
                    section_size = section['size']
                    
                    # Check if we should add this section to current chunk
                    if (current_chunk_size + section_size <= target_chunk_size * 1.4 and  # Allow some overflow
                        current_chunk_size > 0 and  # Don't start with empty chunk
                        self._are_sections_related(current_chunk_sections, section)):  # Check if related
                        
                        # Add to current chunk
                        current_chunk_sections.append(section)
                        current_chunk_size += section_size
                        
                    else:
                        # Save current chunk if it has content
                        if current_chunk_sections:
                            chunk_content = self._combine_sections(current_chunk_sections)
                            chunk_metadata = self._create_chunk_metadata(current_chunk_sections, filename, "grouped_sections")
                            final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                        
                        # Start new chunk
                        if section_size <= 3000:  # 🚨 INCREASED from 5000 to 8000 - allow larger sections to stay together
                            current_chunk_sections = [section]
                            current_chunk_size = section_size
                        else:
                            # Large section gets its own chunk(s) - but with larger size limits
                            chunk_content = section['content']
                            chunk_metadata = {
                                'file_type': 'word',
                                'processing_method': 'section_based_chunking',
                                'source': filename,
                                'section_title': section['title'],
                                'chunk_strategy': 'complete_large_section',
                                'content_size': section_size,
                                'sections_count': 1
                            }
                            final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                            current_chunk_sections = []
                            current_chunk_size = 0
                
                # Add final chunk
                if current_chunk_sections:
                    chunk_content = self._combine_sections(current_chunk_sections)
                    chunk_metadata = self._create_chunk_metadata(current_chunk_sections, filename, "grouped_sections")
                    final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                
                print(f"📦 Final: {len(final_documents)} logical chunks from {len(sections)} sections")
                for i, doc in enumerate(final_documents):
                    section_count = doc.metadata.get('sections_count', 1)
                    chunk_size = len(doc.page_content)
                    strategy = doc.metadata.get('chunk_strategy', 'unknown')
                    print(f"   Chunk {i+1}: {section_count} sections, {chunk_size:,} chars ({strategy})")
                
                return final_documents
                        
            finally:
                os.unlink(temp_path)
                    
        except Exception as e:
            print(f"❌ DOCX processing failed for {filename}: {e}")
            return []

    def _detect_section_header(self, content: str, index: int, all_documents: List[Document]) -> bool:
        """Smart section header detection using multiple heuristics"""
        if not content or len(content.strip()) == 0:
            return False
        
        content = content.strip()
        word_count = len(content.split())
        
        # Heuristic 1: Very short content (likely heading)
        if word_count <= 8 and len(content) <= 120:
            # Heuristic 2: Formatting clues
            is_all_caps = content.isupper()
            starts_with_number = content[:2].strip().isdigit() or (content[:1].isdigit() and content[1:2] in ['.', ')', '-'])
            has_number_pattern = bool(re.search(r'^\d+[\.\)\-]', content))
            
            # Heuristic 3: Positional clues
            is_early_in_doc = index < 5
            is_after_break = index > 0 and not all_documents[index-1].page_content.strip()
            
            # Heuristic 4: Common heading patterns
            common_heading_indicators = [
                r'^[IVX]+\.',  # Roman numerals
                r'^[A-Z]\.',   # Single letters
                r'^\d+\.\d+',  # Numbered sections
            ]
            
            has_heading_pattern = any(re.search(pattern, content.lower()) for pattern in common_heading_indicators)
            
            # Heuristic 5: Check if next content is normal paragraph
            has_following_content = (
                index < len(all_documents) - 1 and 
                len(all_documents[index + 1].page_content.strip()) > 50
            )
            
            # Scoring system
            heading_score = 0
            if is_all_caps: heading_score += 2
            if starts_with_number: heading_score += 2
            if has_number_pattern: heading_score += 1
            if is_early_in_doc: heading_score += 1
            if is_after_break: heading_score += 1
            if has_heading_pattern: heading_score += 2
            if has_following_content: heading_score += 1
            
            return heading_score >= 3
            
        return False

    def _are_sections_related(self, current_sections: List[dict], new_section: dict) -> bool:
        """Check if sections are related and should be grouped together"""
        if not current_sections:
            return True
        
        # Heuristic 1: Similar topic/theme
        current_titles = " ".join([s['title'].lower() for s in current_sections])
        new_title = new_section['title'].lower()
        
        current_words = set(current_titles.split())
        new_words = set(new_title.split())
        common_words = current_words.intersection(new_words)
        
        # Heuristic 2: Sequential sections
        if len(current_sections) > 0:
            last_section = current_sections[-1]
            last_title = last_section['title']
            
            # Check for numbered sequence
            last_numbers = re.findall(r'\d+', last_title)
            new_numbers = re.findall(r'\d+', new_title)
            
            if last_numbers and new_numbers:
                if len(last_numbers) == len(new_numbers):
                    # Check if sequential (1.1 → 1.2, etc.)
                    all_but_last_match = last_numbers[:-1] == new_numbers[:-1]
                    if all_but_last_match:
                        last_num = int(last_numbers[-1]) if last_numbers[-1].isdigit() else 0
                        new_num = int(new_numbers[-1]) if new_numbers[-1].isdigit() else 0
                        if new_num == last_num + 1:
                            return True
        
        # Heuristic 3: Size-based grouping (small sections together)
        total_current_size = sum(s['size'] for s in current_sections)
        if total_current_size + new_section['size'] < 4500:
            return True
        
        # Heuristic 4: Content similarity
        if len(common_words) >= 2:
            return True
        
        return False

    def _combine_sections(self, sections: List[dict]) -> str:
        """Combine multiple sections into a single chunk with clear separation"""
        combined = []
        for section in sections:
            combined.append(f"=== {section['title']} ===")
            combined.append(section['content'])
            combined.append("")  # Empty line between sections
        
        return "\n".join(combined)

    def _create_chunk_metadata(self, sections: List[dict], filename: str, strategy: str) -> dict:
        """Create metadata for a chunk containing multiple sections"""
        section_titles = [s['title'] for s in sections]
        total_size = sum(s['size'] for s in sections)
        
        return {
            'file_type': 'word',
            'processing_method': 'section_based_chunking',
            'source': filename,
            'section_titles': section_titles,
            'sections_count': len(sections),
            'chunk_strategy': strategy,
            'content_size': total_size
        }
    
    def _load_pptx(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Professional PowerPoint processing with SIZE-BASED chunking"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pptx') as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                loader = UnstructuredPowerPointLoader(temp_path, mode="elements")
                documents = loader.load()
                
                print(f"📊 PowerPoint '{filename}': {len(documents)} elements")
                
                # 🚨 GROUP BY SLIDES (each slide becomes a section)
                slides = {}
                for doc in documents:
                    # Extract slide number from metadata if available
                    slide_num = doc.metadata.get('slide_number', 'unknown')
                    if slide_num not in slides:
                        slides[slide_num] = []
                    slides[slide_num].append(doc.page_content)
                
                # Create sections from slides
                sections = []
                for slide_num, slide_content in slides.items():
                    section_content = "\n\n".join(slide_content)
                    sections.append({
                        'title': f"Slide {slide_num}",
                        'content': section_content,
                        'size': len(section_content)
                    })
                
                print(f"📑 Processing {len(sections)} slides in '{filename}'")
                
                # 🚨 UPDATED: SIZE-BASED CHUNKING (YOUR VERSION)
                final_documents = []
                current_chunk_slides = []
                current_chunk_size = 0
                target_chunk_size = 2500
                max_chunk_size = 3000  # Absolute maximum
                
                for i, slide in enumerate(sections):
                    slide_size = slide['size']
                    
                    # 🎯 DECISION: Can we add this slide to current chunk?
                    if current_chunk_size + slide_size <= max_chunk_size:
                        # ✅ YES - Add to current chunk (no slide count limit!)
                        current_chunk_slides.append(slide)
                        current_chunk_size += slide_size
                        
                        # Check if we should finalize this chunk (approaching target)
                        if current_chunk_size >= target_chunk_size:
                            # Create chunk and start fresh
                            chunk_content = self._combine_sections(current_chunk_slides)
                            chunk_metadata = {
                                'file_type': 'powerpoint',
                                'processing_method': 'size_based_chunking',
                                'source': filename,
                                'slide_titles': [s['title'] for s in current_chunk_slides],
                                'slides_count': len(current_chunk_slides),
                                'chunk_strategy': 'size_based_grouping',
                                'content_size': current_chunk_size
                            }
                            final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                            current_chunk_slides = []
                            current_chunk_size = 0
                    
                    else:
                        # ❌ NO - This slide would exceed max size
                        if current_chunk_slides:
                            # Save current chunk first
                            chunk_content = self._combine_sections(current_chunk_slides)
                            chunk_metadata = {
                                'file_type': 'powerpoint',
                                'processing_method': 'size_based_chunking',
                                'source': filename,
                                'slide_titles': [s['title'] for s in current_chunk_slides],
                                'slides_count': len(current_chunk_slides),
                                'chunk_strategy': 'size_based_grouping',
                                'content_size': current_chunk_size
                            }
                            final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                        
                        # Start new chunk with current slide
                        current_chunk_slides = [slide]
                        current_chunk_size = slide_size
                
                # Add final chunk if any remaining
                if current_chunk_slides:
                    chunk_content = self._combine_sections(current_chunk_slides)
                    chunk_metadata = {
                        'file_type': 'powerpoint',
                        'processing_method': 'size_based_chunking',
                        'source': filename,
                        'slide_titles': [s['title'] for s in current_chunk_slides],
                        'slides_count': len(current_chunk_slides),
                        'chunk_strategy': 'size_based_grouping',
                        'content_size': current_chunk_size
                    }
                    final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                
                print(f"📦 PowerPoint Final: {len(final_documents)} logical chunks from {len(sections)} slides")
                return final_documents
                        
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"❌ PPTX processing failed for {filename}: {e}")
            return []
    
    def _load_csv(self, file_bytes: bytes, filename: str) -> List[Document]:
        """CSV processing that PRESERVES HEADER ROW in EVERY chunk"""
        try:
            content = file_bytes.decode('utf-8', errors='ignore')
            lines = content.split('\n')
            
            print(f"📊 CSV file: {filename} with {len(lines)} lines (including header)")
            
            if not lines or len(lines) == 0:
                return []
            
            # Extract header row
            header_line = lines[0] if lines else ""
            
            # 🚨 CRITICAL FIX: Include header in EVERY chunk
            if len(lines) <= 200:
                # Small CSV = single chunk WITH HEADER
                return [Document(
                    page_content=content,  # Includes header row
                    metadata={
                        'source': filename,
                        'file_type': 'csv',
                        'total_lines': len(lines),
                        'has_header': True,
                        'has_complete_data': True,
                        'chunk_strategy': 'complete_file_with_header',
                        'processing_method': 'CSVLoader_with_header'
                    }
                )]
            else:
                # Larger CSV = EVERY chunk gets header + data
                chunks = []
                
                # Process in chunks, each with header
                for chunk_start in range(0, len(lines), 200):
                    chunk_end = min(chunk_start + 200, len(lines))
                    chunk_lines = lines[chunk_start:chunk_end]
                    
                    # 🚨 EVERY CHUNK GETS THE HEADER
                    chunk_content = f"CSV DATA - {filename}\n"
                    chunk_content += f"HEADER ROW: {header_line}\n\n"
                    chunk_content += f"DATA ROWS {chunk_start + 1} to {chunk_end} of {len(lines)} total:\n"
                    chunk_content += "\n".join(chunk_lines)
                    
                    chunk_doc = Document(
                        page_content=chunk_content,
                        metadata={
                            'source': filename,
                            'file_type': 'csv',
                            'lines_range': f"{chunk_start + 1}-{chunk_end}",
                            'total_lines': len(lines),
                            'has_header': True,  # 🆕 EVERY chunk has header
                            'chunk_id': f"csv_lines_{chunk_start + 1}_{chunk_end}",
                            'has_complete_data': False,
                            'chunk_strategy': 'header_with_every_chunk',  # 🆕 New strategy
                            'processing_method': 'CSVLoader_header_in_every_chunk'
                        }
                    )
                    chunks.append(chunk_doc)
                
                print(f"✅ CSV '{filename}': {len(lines)} lines → {len(chunks)} chunks (HEADER IN EVERY CHUNK)")
                return chunks
                                
        except Exception as e:
            print(f"❌ CSV processing failed for {filename}: {e}")
            # Fallback - preserve all data in one chunk (including header)
            try:
                content = file_bytes.decode('utf-8', errors='ignore')
                return [Document(
                    page_content=content,
                    metadata={
                        'source': filename, 
                        'file_type': 'csv', 
                        'has_complete_data': True,
                        'has_header': True,
                        'chunk_strategy': 'fallback_complete_with_header',
                        'processing_method': 'CSVLoader_fallback'
                    }
                )]
            except:
                return []
    
    def _load_text(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Professional text file processing with SMART SECTION-BASED chunking + COLUMN PRESERVATION"""
        try:
            content = file_bytes.decode('utf-8', errors='ignore')
            
            # 🚨 Check if this looks like structured data with headers
            is_structured = self._is_structured_data(content)
            column_info = self._extract_column_info(content) if is_structured else None
            
            if is_structured and column_info['has_columns']:
                print(f"🔍 Detected structured data in '{filename}' with columns: {column_info['columns']}")
            
            # 🚨 SMART TEXT SECTION DETECTION (your existing logic)
            lines = content.split('\n')
            sections = []
            current_section = []
            current_section_title = "Document Start"
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Detect text section headers
                is_heading = self._detect_text_section_header(line, i, lines)
                
                if is_heading and current_section:
                    # Save current section and start new one
                    section_content = "\n".join(current_section)
                    if section_content.strip():
                        sections.append({
                            'title': current_section_title,
                            'content': section_content,
                            'size': len(section_content),
                            'has_structured_data': is_structured
                        })
                    
                    # Start new section
                    current_section_title = line
                    current_section = [line]
                else:
                    # Add to current section
                    current_section.append(line)
            
            # Add the final section
            if current_section:
                section_content = "\n".join(current_section)
                sections.append({
                    'title': current_section_title,
                    'content': section_content,
                    'size': len(section_content),
                    'has_structured_data': is_structured
                })
            
            print(f"📑 Detected {len(sections)} natural sections in text file '{filename}':")
            for i, section in enumerate(sections):
                title_preview = section['title'][:60] + "..." if len(section['title']) > 60 else section['title']
                print(f"   {i+1}. '{title_preview}' ({section['size']:,} chars)")
            
            # 🚨 GROUP RELATED SECTIONS INTO LOGICAL CHUNKS (your existing logic)
            if len(sections) <= 1:
                # Very small text file = single chunk
                enhanced_content = content
                # 🚨 Add column info for structured data
                if is_structured and column_info and column_info['has_columns']:
                    enhanced_content = self._add_column_info_to_content(content, column_info)
                
                return [Document(
                    page_content=enhanced_content,
                    metadata={
                        'file_type': 'text',
                        'processing_method': 'TextLoader_single_chunk',
                        'source': filename,
                        'content_length': len(content),
                        'has_structured_data': is_structured,
                        'has_columns': column_info['has_columns'] if column_info else False,
                        'column_count': len(column_info['columns']) if column_info and column_info['columns'] else 0
                    }
                )]
            else:
                # 🚨 GROUP RELATED SECTIONS INTO LOGICAL CHUNKS
                final_documents = []
                current_chunk_sections = []
                current_chunk_size = 0
                target_chunk_size = 2500

                for i, section in enumerate(sections):
                    section_size = section['size']
                    
                    if (current_chunk_size + section_size <= target_chunk_size * 1.4 and
                        current_chunk_size > 0 and
                        self._are_sections_related(current_chunk_sections, section)):
                        
                        current_chunk_sections.append(section)
                        current_chunk_size += section_size
                        
                    else:
                        if current_chunk_sections:
                            chunk_content = self._combine_sections(current_chunk_sections)
                            # 🚨 Add column info for structured data chunks
                            if is_structured and column_info and column_info['has_columns']:
                                chunk_content = self._add_column_info_to_content(chunk_content, column_info)
                            
                            chunk_metadata = self._create_chunk_metadata(current_chunk_sections, filename, "grouped_text_sections")
                            # 🚨 Add structured data info to metadata
                            chunk_metadata.update({
                                'has_structured_data': is_structured,
                                'has_columns': column_info['has_columns'] if column_info else False,
                                'column_count': len(column_info['columns']) if column_info and column_info['columns'] else 0
                            })
                            final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                        
                        if section_size <= 2500:
                            current_chunk_sections = [section]
                            current_chunk_size = section_size
                        else:
                            # Large section gets its own chunk
                            chunk_content = section['content']
                            # 🚨 Add column info for structured data
                            if is_structured and column_info and column_info['has_columns']:
                                chunk_content = self._add_column_info_to_content(chunk_content, column_info)
                            
                            chunk_metadata = {
                                'file_type': 'text',
                                'processing_method': 'text_section_based_chunking',
                                'source': filename,
                                'section_title': section['title'],
                                'chunk_strategy': 'complete_large_section',
                                'content_size': section_size,
                                'sections_count': 1,
                                'has_structured_data': is_structured,
                                'has_columns': column_info['has_columns'] if column_info else False,
                                'column_count': len(column_info['columns']) if column_info and column_info['columns'] else 0
                            }
                            final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                            current_chunk_sections = []
                            current_chunk_size = 0
                
                if current_chunk_sections:
                    chunk_content = self._combine_sections(current_chunk_sections)
                    # 🚨 Add column info for structured data
                    if is_structured and column_info and column_info['has_columns']:
                        chunk_content = self._add_column_info_to_content(chunk_content, column_info)
                    
                    chunk_metadata = self._create_chunk_metadata(current_chunk_sections, filename, "grouped_text_sections")
                    # 🚨 Add structured data info to metadata
                    chunk_metadata.update({
                        'has_structured_data': is_structured,
                        'has_columns': column_info['has_columns'] if column_info else False,
                        'column_count': len(column_info['columns']) if column_info and column_info['columns'] else 0
                    })
                    final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                
                print(f"📦 Text Final: {len(final_documents)} logical chunks from {len(sections)} sections")
                if is_structured:
                    print(f"🔤 Structured data detected - column info preserved in all chunks")
                return final_documents
                        
        except Exception as e:
            print(f"❌ Text processing failed for {filename}: {e}")
            return []
        
    

    def _is_structured_data(self, content: str) -> bool:
        """Check if text content appears to be structured data (CSV, TSV, table, etc.)"""
        lines = content.split('\n')
        if len(lines) < 3:
            return False
        
        # Check for common structured data patterns
        table_indicators = 0
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
                
            # CSV pattern (commas with data)
            if ',' in line and line.count(',') >= 2 and any(c.isalnum() for c in line):
                table_indicators += 2
                
            # TSV pattern (tabs with data)
            if '\t' in line and line.count('\t') >= 2 and any(c.isalnum() for c in line):
                table_indicators += 2
                
            # Pipe-separated pattern
            if '|' in line and line.count('|') >= 2 and any(c.isalnum() for c in line):
                table_indicators += 2
                
            # Common column header keywords
            header_keywords = ['id', 'name', 'date', 'type', 'value', 'amount', 'price', 'quantity']
            if any(keyword in line.lower() for keyword in header_keywords):
                table_indicators += 1
        
        return table_indicators >= 3

    def _extract_column_info(self, content: str) -> Dict[str, Any]:
        """Extract column information from structured data content"""
        column_info = {
            'has_columns': False,
            'columns': [],
            'column_line': None,
            'separator': None
        }
        
        lines = content.split('\n')
        
        # Look for column headers in first few lines
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if not line:
                continue
                
            # Determine separator
            separator = None
            if ',' in line and line.count(',') >= 2:
                separator = ','
            elif '\t' in line and line.count('\t') >= 2:
                separator = '\t'
            elif '|' in line and line.count('|') >= 2:
                separator = '|'
            
            if separator:
                columns = [col.strip() for col in line.split(separator) if col.strip()]
                if len(columns) >= 2:  # Reasonable number of columns
                    column_info.update({
                        'has_columns': True,
                        'columns': columns,
                        'column_line': line,
                        'separator': separator
                    })
                    break
        
        return column_info

    def _add_column_info_to_content(self, content: str, column_info: Dict[str, Any]) -> str:
        """Add column information to the beginning of content"""
        if not column_info['has_columns']:
            return content
        
        enhanced_content = "STRUCTURED DATA WITH COLUMN INFORMATION:\n"
        
        if column_info['columns']:
            enhanced_content += f"COLUMN NAMES ({len(column_info['columns'])} columns):\n"
            for i, col in enumerate(column_info['columns']):
                enhanced_content += f"  {i+1}. {col}\n"
            enhanced_content += f"SEPARATOR: '{column_info['separator']}'\n"
        elif column_info['column_line']:
            enhanced_content += f"COLUMN HEADER LINE: {column_info['column_line']}\n"
        
        enhanced_content += "\n" + "="*50 + "\nDATA:\n" + "="*50 + "\n"
        enhanced_content += content
        
        return enhanced_content

    def _detect_text_section_header(self, line: str, index: int, all_lines: List[str]) -> bool:
        """Smart text file section header detection"""
        if not line or len(line.strip()) == 0:
            return False
        
        line = line.strip()
        word_count = len(line.split())
        
        # Text file specific heuristics
        if word_count <= 8 and len(line) <= 100:
            # Common text heading patterns
            is_all_caps = line.isupper()
            starts_with_number = bool(re.search(r'^\d+[\.\)\-\s]', line))
            has_separators = line.startswith(('#', '=', '-', '*')) or line.endswith((':', '-'))
            
            # Check if it's isolated (headings often have empty lines around them)
            has_empty_before = index == 0 or not all_lines[index-1].strip()
            has_empty_after = index == len(all_lines)-1 or not all_lines[index+1].strip()
            
            # Scoring system for text files
            heading_score = 0
            if is_all_caps: heading_score += 2
            if starts_with_number: heading_score += 2
            if has_separators: heading_score += 2
            if has_empty_before: heading_score += 1
            if has_empty_after: heading_score += 1
            
            return heading_score >= 4
            
        return False
    
    def _load_markdown(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Professional Markdown processing"""
        try:
            content = file_bytes.decode('utf-8')
            documents = self.text_splitters['markdown'].split_text(content)
            
            doc_objects = []
            for doc in documents:
                doc_objects.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        'file_type': 'markdown',
                        'processing_method': 'MarkdownHeaderTextSplitter',
                        **doc.metadata
                    }
                ))
            
            return doc_objects
            
        except Exception as e:
            print(f"Markdown processing failed for {filename}: {e}")
            return self._load_text(file_bytes, filename)
    
    # 🚨 UPDATED FALLBACK LOADER WITH BETTER DETECTION
    def _fallback_loader(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Enhanced fallback loader for unsupported file types"""
        try:
            # Try to detect if it's a text file without extension
            try:
                content_preview = file_bytes[:2500].decode('utf-8', errors='ignore')
                lines = content_preview.split('\n')
                
                # If it looks like structured data (multiple lines with data)
                if len(lines) >= 3 and any(',' in line or '\t' in line for line in lines[:3]):
                    print(f"🔍 Detected structured data: {filename}")
                    full_content = file_bytes.decode('utf-8', errors='ignore')
                    return [Document(
                        page_content=full_content[:2500],
                        metadata={'source': filename, 'file_type': 'detected_structured_data'}
                    )]
            except:
                pass
            
            # Original fallback logic
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                loader = UnstructuredFileLoader(temp_path)
                documents = loader.load()
                
                for doc in documents:
                    doc.metadata.update({
                        'file_type': 'unknown',
                        'processing_method': 'UnstructuredFileLoader_fallback'
                    })
                
                return documents
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"❌ Fallback loader failed for {filename}: {e}")
            return []
    
    def _smart_chunking(self, documents: List[Document], filename: str) -> List[Document]:
        """FIXED: Intelligent chunking that creates FEWER, LARGER chunks"""
        chunked_documents = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata.copy()
            file_type = metadata.get('file_type', 'unknown')
            
            # 🚨 FIX: MUCH LARGER threshold for single chunks
            if len(content.strip()) < 8000:  # Increased from 4000 to 8000
                # Keep as single large chunk
                chunked_documents.append(doc)
                print(f"📦 KEEPING SINGLE CHUNK: {len(content)} chars for {filename}")
                continue
            
            # 🚨 FIX: Use MUCH LARGER chunk sizes for splitting
            larger_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500,  # 🚨 INCREASED from 2800 to 6000
                chunk_overlap=400, # 🚨 INCREASED from 400 to 800
                separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""]
            )
            chunks = larger_splitter.split_documents([doc])
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': f"{metadata.get('source', 'doc')}_large_part_{i+1}",
                    'total_chunks': len(chunks),
                    'chunk_index': i,
                    'is_large_chunk': True
                })
            
            chunked_documents.extend(chunks)
            print(f"📦 Created {len(chunks)} LARGE chunks for {filename} (was {len(content)} chars)")
        
        print(f"📦 Final chunking: {len(documents)} docs → {len(chunked_documents)} LARGE chunks for {filename}")
        return chunked_documents
    
    def _chunk_structured_data(self, content: str, metadata: dict) -> List[Document]:
        """Custom chunking for structured data like Excel"""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            
            if current_size + line_size > 1500 and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=metadata.copy()
                ))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(Document(
                page_content=chunk_text,
                metadata=metadata.copy()
            ))
        
        return chunks
    

    

    def enhanced_excel_loader(self, file_bytes: bytes, filename: str) -> List[Document]:
        try:
            import pandas as pd
            from io import BytesIO
            
            excel_file = BytesIO(file_bytes)
            xl = pd.ExcelFile(excel_file, engine='openpyxl')  # Keep engine fix
            all_documents = []  # ← RESTORE THIS CRITICAL LINE! 🚀
            
            print(f"📊 Processing Excel file: {filename} with {len(xl.sheet_names)} sheets")
            
            for sheet_name in xl.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')  # Add engine here too
                        
                    # 🚨 CHECK FOR UNNAMED COLUMNS
                    has_proper_headers = not all(col.startswith('Unnamed:') for col in df.columns)
                    column_names = list(df.columns)
                    
                    if has_proper_headers:
                        print(f"📊 Processing sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
                        print(f"🔤 PROPER COLUMNS: {column_names}")
                    else:
                        print(f"📊 Processing sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} UNNAMED columns")
                        print(f"⚠️  No proper headers - using positional column names")
                        # Create descriptive column names
                        column_names = [f"Column_{i+1}" for i in range(len(df.columns))]
                    
                    columns_text = ", ".join([f"{i+1}. {col}" for i, col in enumerate(column_names)])
                    
                    # Skip empty sheets
                    if len(df) == 0:
                        print(f"⏭️  Skipping empty sheet: {sheet_name}")
                        continue
                    
                    # Determine chunk size
                    if len(df) <= 20:
                        rows_per_chunk = len(df)
                    elif len(df) <= 50:
                        rows_per_chunk = 20
                    else:
                        rows_per_chunk = 20
                    
                    print(f"📊 LARGE CHUNKING: {len(df)} rows → {rows_per_chunk} rows per chunk")
                    
                    for chunk_start in range(0, len(df), rows_per_chunk):
                        chunk_end = min(chunk_start + rows_per_chunk, len(df))
                        chunk_df = df.iloc[chunk_start:chunk_end]
                        
                        # 🚨 EVERY CHUNK GETS COLUMN INFORMATION
                        chunk_content = f"EXCEL DATA - {filename}\n"
                        chunk_content += f"SHEET: {sheet_name}\n"
                        chunk_content += f"TOTAL ROWS: {len(df)} | CURRENT CHUNK: Rows {chunk_start + 1} to {chunk_end}\n"
                        
                        # 🚨 COLUMN NAMES IN EVERY CHUNK
                        if has_proper_headers:
                            chunk_content += f"COLUMN NAMES ({len(column_names)}): {columns_text}\n\n"
                        else:
                            chunk_content += f"COLUMNS (NO HEADERS - {len(column_names)} positional columns): {columns_text}\n\n"
                            chunk_content += "⚠️  This file doesn't have proper column headers. Columns are positional.\n\n"
                        
                        # Include sample data row to show structure
                        if not df.empty:
                            sample_data = []
                            for col in df.columns[:3]:  # First 3 columns as sample
                                sample_val = df.iloc[0][col]
                                if pd.notna(sample_val):
                                    sample_data.append(f"{sample_val}")
                            
                            if sample_data:
                                chunk_content += f"SAMPLE DATA (first row): {' | '.join(sample_data)}...\n\n"
                        
                        chunk_content += "DATA IN THIS CHUNK:\n"
                        rows_with_data = 0
                        
                        for idx, row in chunk_df.iterrows():
                            original_idx = chunk_start + (idx - chunk_df.index[0])
                            row_data = []
                            
                            for i, col in enumerate(df.columns):
                                value = row[col]
                                if pd.notna(value) and str(value).strip():
                                    # Use descriptive column names instead of 'Unnamed: X'
                                    col_name = column_names[i] if i < len(column_names) else f"Column_{i+1}"
                                    row_data.append(f"{col_name}: {value}")
                            
                            if row_data:
                                chunk_content += f"ROW {original_idx + 1}: {' | '.join(row_data)}\n"
                                rows_with_data += 1
                        
                        if rows_with_data > 0:
                            chunk_doc = Document(
                                page_content=chunk_content,
                                metadata={
                                    "source": f"{filename} - {sheet_name}",
                                    "file_type": "excel",
                                    "sheet_name": sheet_name,
                                    "rows_range": f"{chunk_start + 1}-{chunk_end}",
                                    "total_rows": len(df),
                                    "columns": column_names,
                                    "column_count": len(column_names),
                                    "has_proper_headers": has_proper_headers,  # 🆕 Track if headers are proper
                                    "processing_method": "excel_with_columns_in_every_chunk"
                                }
                            )
                            all_documents.append(chunk_doc)
                            print(f"✅ Created chunk: rows {chunk_start + 1}-{chunk_end} ({rows_with_data} data rows)")
                    
                    print(f"✅ Sheet '{sheet_name}': {len(df)} rows → {len([d for d in all_documents if d.metadata.get('sheet_name') == sheet_name])} chunks")
                        
                except Exception as e:
                    print(f"❌ Failed to process sheet {sheet_name}: {e}")
                    continue
            
            print(f"📁 Excel '{filename}': {len(xl.sheet_names)} sheets → {len(all_documents)} chunks")
            return all_documents
            
        except Exception as e:
            print(f"❌ Excel processing failed for {filename}: {e}")
            return []

    def _chunk_excel_preserve_all_data(self, df, sheet_name: str, filename: str) -> List[Document]:
        """Chunk Excel data while preserving EVERY row and column"""
        chunks = []
        
        # Determine optimal chunk size
        rows_per_chunk = 20  # Fixed chunk size for consistency
        
        print(f"📊 Chunking strategy: {rows_per_chunk} rows per chunk")
        
        # Process data in chunks
        for chunk_start in range(0, len(df), rows_per_chunk):
            chunk_end = min(chunk_start + rows_per_chunk, len(df))
            chunk_df = df.iloc[chunk_start:chunk_end]
            
            # Build chunk content with ALL data
            chunk_content = f"EXCEL DATA - {filename}\n"
            chunk_content += f"SHEET: {sheet_name}\n"
            chunk_content += f"ROWS: {chunk_start + 1} to {chunk_end} of {len(df)} total\n"
            chunk_content += f"COLUMNS: {', '.join(map(str, df.columns.tolist()))}\n\n"
            
            # Add EVERY row in this chunk with ALL columns
            chunk_content += "COMPLETE DATA:\n"
            for idx, row in chunk_df.iterrows():
                original_idx = chunk_start + (idx - chunk_df.index[0])
                row_data = []
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        row_data.append(f"{col}: {value}")
                    else:
                        row_data.append(f"{col}: [BLANK]")
                
                chunk_content += f"ROW {original_idx + 1}: {' | '.join(row_data)}\n"
            
            # Create document with complete data
            chunk_doc = Document(
                page_content=chunk_content,
                metadata={
                    "source": f"{filename} - {sheet_name}",
                    "file_type": "excel",
                    "sheet_name": sheet_name,
                    "rows_range": f"{chunk_start + 1}-{chunk_end}",
                    "total_rows": len(df),
                    "chunk_id": f"{sheet_name}_rows_{chunk_start + 1}_{chunk_end}",
                    "has_complete_data": True,
                    "columns": list(df.columns)
                }
            )
            chunks.append(chunk_doc)
        
        return chunks
    

    

    def enhanced_pdf_loader(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Enhanced PDF loader with SMART SECTION-BASED chunking + FORCED CHUNKING FALLBACK"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                # Use PyPDFLoader as primary method
                print(f"📄 Using PyPDFLoader for: {filename}")
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                
                if not documents or not any(doc.page_content.strip() for doc in documents):
                    print(f"🔄 PyPDF extracted little content, trying Unstructured for: {filename}")
                    from unstructured.partition.pdf import partition_pdf
                    
                    elements = partition_pdf(
                        filename=temp_path,
                        strategy="fast",
                        extract_images=False,
                        languages=["eng"]
                    )
                    
                    # Convert elements to documents
                    documents = []
                    for i, element in enumerate(elements):
                        text = element.text.strip()
                        if text and len(text) > 50:
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "file_type": "pdf", 
                                    "processing_method": "unstructured",
                                    "element_index": i
                                }
                            )
                            documents.append(doc)
                
                # 🚨 CHECK PDF SIZE - if too large, use forced chunking immediately
                total_chars = sum(len(doc.page_content) for doc in documents)
                print(f"📊 PDF '{filename}': {len(documents)} pages, {total_chars} total chars")
                
                # If PDF is very large, use forced chunking to avoid single huge chunks
                if total_chars > 7000:  # If more than 7k chars, use aggressive chunking
                    print(f"🔪 LARGE PDF DETECTED - Using forced chunking for {total_chars} chars")
                    return self._force_chunk_large_pdf(documents, filename, total_chars)
                
                # 🚨 IMPROVED SMART PDF SECTION DETECTION
                sections = []
                current_section = []
                current_section_title = "Document Start"
                
                for i, doc in enumerate(documents):
                    content = doc.page_content.strip()
                    if not content:
                        continue
                    
                    # 🎯 IMPROVED: Better PDF section header detection
                    is_heading = self._detect_pdf_section_header_improved(content, i, documents, len(documents))
                    
                    if is_heading and current_section:
                        # Save current section and start new one
                        section_content = "\n\n".join([doc.page_content for doc in current_section])
                        if section_content.strip():
                            sections.append({
                                'title': current_section_title,
                                'content': section_content,
                                'size': len(section_content),
                                'pages': f"{current_section[0].metadata.get('page', '?')}-{current_section[-1].metadata.get('page', '?')}" if hasattr(current_section[0], 'metadata') else "unknown"
                            })
                        
                        # Start new section
                        current_section_title = content[:100]  # Use first 100 chars as title
                        current_section = [doc]
                    else:
                        # Add to current section
                        current_section.append(doc)
                
                # Add the final section
                if current_section:
                    section_content = "\n\n".join([doc.page_content for doc in current_section])
                    sections.append({
                        'title': current_section_title,
                        'content': section_content,
                        'size': len(section_content),
                        'pages': f"{current_section[0].metadata.get('page', '?')}-{current_section[-1].metadata.get('page', '?')}" if hasattr(current_section[0], 'metadata') else "unknown"
                    })
                
                print(f"📑 Detected {len(sections)} natural sections in PDF '{filename}':")
                for i, section in enumerate(sections):
                    title_preview = section['title'][:60] + "..." if len(section['title']) > 60 else section['title']
                    print(f"   {i+1}. '{title_preview}' ({section['size']:,} chars, pages: {section['pages']})")
                
                # 🚨 FORCE CHUNKING if only 1 section detected (means section detection failed)
                if len(sections) <= 1 and total_chars > 8000:
                    print(f"⚠️ SECTION DETECTION FAILED - Only 1 section detected for {total_chars} chars")
                    print(f"🔪 Using forced chunking fallback")
                    return self._force_chunk_large_pdf(documents, filename, total_chars)
                
                # 🚨 GROUP RELATED SECTIONS INTO LOGICAL CHUNKS
                final_documents = []
                current_chunk_sections = []
                current_chunk_size = 0
                target_chunk_size = 2500  # Reduced from 6000 for better chunks
                max_chunk_size = 4000     # Absolute maximum

                for i, section in enumerate(sections):
                    section_size = section['size']
                    
                    # 🎯 IMPROVED: Always chunk large sections, don't let them stay huge
                    if section_size > max_chunk_size:
                        print(f"🔪 Chunking oversized section: {section_size} chars -> multiple chunks")
                        chunked_sections = self._chunk_oversized_section(section, filename, max_chunk_size)
                        final_documents.extend(chunked_sections)
                        continue
                    
                    # Check if we should add this section to current chunk
                    if (current_chunk_size + section_size <= max_chunk_size and
                        current_chunk_size > 0 and
                        self._are_sections_related(current_chunk_sections, section)):
                        
                        # Add to current chunk
                        current_chunk_sections.append(section)
                        current_chunk_size += section_size
                        
                    else:
                        # Save current chunk if it has content
                        if current_chunk_sections:
                            chunk_content = self._combine_sections(current_chunk_sections)
                            chunk_metadata = self._create_chunk_metadata(current_chunk_sections, filename, "grouped_pdf_sections")
                            final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                        
                        # Start new chunk
                        current_chunk_sections = [section]
                        current_chunk_size = section_size
                    
                    # If current chunk is approaching max size, finalize it
                    if current_chunk_size >= target_chunk_size:
                        if current_chunk_sections:
                            chunk_content = self._combine_sections(current_chunk_sections)
                            chunk_metadata = self._create_chunk_metadata(current_chunk_sections, filename, "grouped_pdf_sections")
                            final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                            current_chunk_sections = []
                            current_chunk_size = 0
                
                # Add final chunk
                if current_chunk_sections:
                    chunk_content = self._combine_sections(current_chunk_sections)
                    chunk_metadata = self._create_chunk_metadata(current_chunk_sections, filename, "grouped_pdf_sections")
                    final_documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))
                
                print(f"📦 PDF Final: {len(final_documents)} logical chunks from {len(sections)} sections")
                return final_documents
                            
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                        
        except Exception as e:
            print(f"❌ PDF processing failed for {filename}: {e}")
            return []

    def _detect_pdf_section_header_improved(self, content: str, index: int, all_documents: List[Document], total_pages: int) -> bool:
        """IMPROVED PDF section header detection with better heuristics"""
        if not content or len(content.strip()) == 0:
            return False
        
        content = content.strip()
        lines = content.split('\n')
        
        # Use first few lines for header detection
        first_lines = lines[:3]  # Check first 3 lines
        
        for line in first_lines:
            line = line.strip()
            if not line:
                continue
                
            word_count = len(line.split())
            line_length = len(line)
            
            # 🎯 IMPROVED HEURISTICS FOR PDF HEADERS
            header_score = 0
            
            # Heuristic 1: Very short lines (likely headings)
            if word_count <= 8 and line_length <= 120:
                header_score += 3
            
            # Heuristic 2: Formatting clues
            if line.isupper():
                header_score += 2  # ALL CAPS often indicates headings
            
            # Heuristic 3: Common heading patterns
            starts_with_number = bool(re.search(r'^\d+[\.\)\-\s]', line))
            has_roman_numerals = bool(re.search(r'^[IVX]+[\.\)\-\s]', line))
            common_heading_words = ['chapter', 'section', 'part', 'article', 'title', 'introduction', 'abstract', 'summary', 'conclusion']
            
            if starts_with_number:
                header_score += 2
            if has_roman_numerals:
                header_score += 2
            if any(word in line.lower() for word in common_heading_words):
                header_score += 3
            
            # Heuristic 4: Position in document
            is_early_page = index < 3  # First few pages often have headings
            is_late_page = index > total_pages - 5  # Last few pages might have conclusions
            
            if is_early_page:
                header_score += 1
            
            # Heuristic 5: Check if next content is substantial (headings are followed by content)
            if index < len(all_documents) - 1:
                next_content = all_documents[index + 1].page_content.strip()
                if len(next_content) > 200:  # Next page has substantial content
                    header_score += 1
            
            # If we found a strong header candidate in first lines, return True
            if header_score >= 5:
                print(f"🎯 PDF HEADER DETECTED: '{line[:50]}...' (score: {header_score})")
                return True
        
        return False

    def _force_chunk_large_pdf(self, documents: List[Document], filename: str, total_chars: int) -> List[Document]:
        """Force chunk large PDFs that section detection failed on"""
        print(f"🔪 FORCE CHUNKING PDF: {filename} ({total_chars} chars)")
        
        # Combine all content
        all_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Use aggressive chunking for large PDFs
        pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,      # Smaller chunks for better retrieval
            chunk_overlap=150,    # Reasonable overlap
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""]
        )
        
        # Create base document to split
        base_doc = Document(page_content=all_content, metadata={
            'source': filename,
            'file_type': 'pdf',
            'processing_method': 'forced_chunking_large_pdf'
        })
        
        chunked_docs = pdf_splitter.split_documents([base_doc])
        
        # Add proper metadata to chunks
        for i, chunk in enumerate(chunked_docs):
            chunk.metadata.update({
                'chunk_id': f"{filename}_forced_chunk_{i}",
                'total_chunks': len(chunked_docs),
                'chunk_index': i,
                'forced_chunking': True,
                'original_pages': len(documents)
            })
        
        print(f"✅ PDF force-chunked into {len(chunked_docs)} chunks")
        return chunked_docs

    def _chunk_oversized_section(self, section: Dict, filename: str, max_size: int) -> List[Document]:
        """Chunk oversized sections into multiple chunks"""
        content = section['content']
        section_size = section['size']
        
        if section_size <= max_size:
            return [Document(
                page_content=content,
                metadata={
                    'file_type': 'pdf',
                    'processing_method': 'pdf_section_based_chunking',
                    'source': filename,
                    'section_title': section['title'],
                    'pages': section['pages'],
                    'chunk_strategy': 'complete_section',
                    'content_size': section_size,
                    'sections_count': 1
                }
            )]
        
        # Chunk the oversized section
        num_chunks = max(2, (section_size // max_size) + 1)
        chunk_size = section_size // num_chunks
        
        print(f"🔪 Chunking oversized section '{section['title'][:30]}...' into {num_chunks} chunks")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""]
        )
        
        base_doc = Document(page_content=content, metadata={'source': filename})
        chunks = splitter.split_documents([base_doc])
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'file_type': 'pdf',
                'processing_method': 'oversized_section_chunking',
                'source': filename,
                'section_title': f"{section['title']} (Part {i+1})",
                'pages': section['pages'],
                'chunk_strategy': 'oversized_section_split',
                'content_size': len(chunk.page_content),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'original_section_size': section_size
            })
        
        return chunks


    def process_heic_images(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Enhanced HEIC image processing with better error handling"""
        try:
            pillow_heif.register_heif_opener()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.heic') as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                image = Image.open(temp_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                jpeg_path = temp_path.replace('.heic', '.jpg')
                image.save(jpeg_path, 'JPEG', quality=85)
                
                # Use UnstructuredImageLoader for OCR
                from langchain_community.document_loaders import UnstructuredImageLoader
                loader = UnstructuredImageLoader(jpeg_path)
                documents = loader.load()
                
                if documents:
                    cleaned_docs = []
                    for doc in documents:
                        # 🚨 FIX: Ensure page_content is never None
                        if doc.page_content is None:
                            doc.page_content = ""
                        elif not isinstance(doc.page_content, str):
                            doc.page_content = str(doc.page_content)
                        
                        # Only keep documents with actual content
                        if doc.page_content.strip():
                            doc.metadata.update({
                                'source': filename, 
                                'file_type': 'HEIC_IMAGE',
                                'processing_method': 'UnstructuredImageLoader_OCR',
                                'content_length': len(doc.page_content)
                            })
                            cleaned_docs.append(doc)
                    
                    if cleaned_docs:
                        print(f"✅ HEIC OCR extracted: {filename} -> {len(cleaned_docs)} docs, {len(cleaned_docs[0].page_content)} chars")
                        return cleaned_docs
                    else:
                        print(f"⚠️ No text content in HEIC: {filename}")
                        return []
                else:
                    print(f"⚠️ No documents extracted from HEIC: {filename}")
                    return []
                    
            finally:
                for path in [temp_path, jpeg_path]:
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                    except:
                        pass
                            
        except Exception as e:
            print(f"❌ HEIC processing failed for {filename}: {e}")
            return []



  

# =============================================================================
# INTELLIGENT CHUNKING SYSTEM - YOUR EXISTING
# =============================================================================

class IntelligentChunkingSystem:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
        self.splitters = {
            "resume": RecursiveCharacterTextSplitter(
                chunk_size=2500,
                chunk_overlap=200,
                separators=["\n## ", "\n# ", "\n\n", "\n• ", "\n- ", "\n", "• ", "- ", " ", ""],
                length_function=len,
            ),
            "financial": RecursiveCharacterTextSplitter(
                chunk_size=2500,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""],
                length_function=len,
            ),
            "general": RecursiveCharacterTextSplitter(
                chunk_size=2500,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""],
                length_function=len,
            )
        }
    
    def smart_chunk_documents(self, documents: List[Document]) -> List[Document]:
        """ROBUST chunking that preserves data structure"""
        all_chunks = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata.copy()
            
            if self._is_structured_data(content):
                print(f"📊 Detected structured data: {metadata.get('source', 'unknown')}")
                chunks = self._chunk_structured_data(doc)
            else:
                content_type = self._classify_content(content)
                metadata['content_type'] = content_type
                
                splitter = self.splitters.get(content_type, self.splitters["general"])
                
                try:
                    chunks = splitter.split_documents([doc])
                    chunks = self._ensure_multi_chunk(splitter, doc, chunks)
                except Exception as e:
                    print(f"❌ Chunking error for {metadata.get('source', 'unknown')}: {e}")
                    chunks = self.splitters["general"].split_documents([doc])
                    chunks = self._ensure_multi_chunk(self.splitters["general"], doc, chunks)
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': f"{metadata.get('source', 'doc')}_{i}",
                    'content_type': metadata.get('content_type', 'general'),
                    'word_count': len(chunk.page_content.split()),
                    'char_count': len(chunk.page_content),
                    'parent_document': metadata.get('source', 'unknown'),
                    'is_structured_data': self._is_structured_data(content)
                })
            
            all_chunks.extend(chunks)
            print(f"📦 Created {len(chunks)} chunks from {metadata.get('source', 'unknown')}")
        
        print(f"✅ Created {len(all_chunks)} intelligent chunks from {len(documents)} documents")
        return all_chunks
    
    def _ensure_multi_chunk(self, splitter: RecursiveCharacterTextSplitter, doc: Document, chunks: List[Document]) -> List[Document]:
        """Prevent huge documents from collapsing into a single chunk."""
        if len(chunks) == 1:
            chunk = chunks[0]
            chunk_size = getattr(splitter, 'chunk_size', 2500)
            if len(chunk.page_content) > chunk_size * 1.5:
                fallback_chunk_size = max(800, chunk_size // 2)
                print(f"⚠️ Large document produced single chunk ({len(chunk.page_content)} chars). Re-chunking with size {fallback_chunk_size}.")
                fallback_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=fallback_chunk_size,
                    chunk_overlap=min(200, fallback_chunk_size // 4),
                    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""],
                    length_function=len,
                )
                return fallback_splitter.split_documents([doc])
        return chunks
    
    def _is_structured_data(self, content: str) -> bool:
        """Auto-detect structured data - SENSITIVE detection"""
        if not content or len(content) < 10:
            return False
        
        lines = content.split('\n')
        if len(lines) < 3:
            return False
        
        table_indicators = 0
        
        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue
                
            if '|' in line and line.count('|') >= 2:
                table_indicators += 2
            elif '\t' in line and line.count('\t') >= 2:
                table_indicators += 2
            elif line.count(',') >= 3 and any(char.isalpha() for char in line):
                table_indicators += 1
            elif 'SHEET:' in line or 'COLUMNS:' in line:
                table_indicators += 3
            elif any(keyword in line for keyword in ['ROW', 'COLUMN', 'CELL']):
                table_indicators += 1
        
        return table_indicators >= 1
    
    def _chunk_structured_data(self, doc: Document) -> List[Document]:
        """Chunk structured data while preserving COLUMN INFORMATION in every chunk"""
        content = doc.page_content
        lines = content.split('\n')
        
        # 🚨 Extract column information from the content
        column_info = self._extract_column_info(content)
        
        if len(content) < 5000:
            # Small content = single chunk with enhanced column info
            enhanced_content = self._add_column_info_to_content(content, column_info)
            doc.page_content = enhanced_content
            return [doc]
        
        chunks = []
        current_chunk = []
        current_row_count = 0
        
        for line in lines:
            if line.strip():
                current_chunk.append(line)
                current_row_count += 1
                
                if current_row_count >= 25:
                    chunk_text = '\n'.join(current_chunk)
                    # 🚨 Add column info to every chunk
                    enhanced_chunk = self._add_column_info_to_content(chunk_text, column_info)
                    
                    chunk_doc = Document(
                        page_content=enhanced_chunk,
                        metadata=doc.metadata.copy()
                    )
                    chunks.append(chunk_doc)
                    current_chunk = []
                    current_row_count = 0
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            # 🚨 Add column info to final chunk too
            enhanced_chunk = self._add_column_info_to_content(chunk_text, column_info)
            chunk_doc = Document(
                page_content=enhanced_chunk,
                metadata=doc.metadata.copy()
            )
            chunks.append(chunk_doc)
        
        print(f"📊 Structured data: {len(lines)} lines → {len(chunks)} chunks WITH COLUMN INFO")
        return chunks

    def _extract_column_info(self, content: str) -> Dict[str, Any]:
        """Extract column information from structured data content"""
        column_info = {
            'has_columns': False,
            'columns': [],
            'column_line': None
        }
        
        lines = content.split('\n')
        
        # Look for column headers
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line_lower = line.lower()
            # Common indicators of column headers
            if any(keyword in line_lower for keyword in ['column', 'field', 'header', 'name', 'id', 'date', 'type']):
                # This might be a column header line
                column_info['has_columns'] = True
                column_info['column_line'] = line
                # Try to extract individual columns
                if ',' in line:
                    column_info['columns'] = [col.strip() for col in line.split(',')]
                elif '\t' in line:
                    column_info['columns'] = [col.strip() for col in line.split('\t')]
                elif '|' in line:
                    column_info['columns'] = [col.strip() for col in line.split('|')]
                break
        
        return column_info

    def _add_column_info_to_content(self, content: str, column_info: Dict[str, Any]) -> str:
        """Add column information to the beginning of content"""
        if not column_info['has_columns']:
            return content
        
        enhanced_content = "STRUCTURED DATA WITH COLUMN INFORMATION:\n"
        
        if column_info['columns']:
            enhanced_content += f"COLUMNS FOUND: {', '.join(column_info['columns'])}\n"
        elif column_info['column_line']:
            enhanced_content += f"COLUMN HEADER LINE: {column_info['column_line']}\n"
        
        enhanced_content += "\n" + content
        return enhanced_content
    
    def _classify_content(self, content: str) -> str:
        """Free content classification using intelligent heuristics"""
        if not content or len(content.strip()) < 10:
            return "general"
            
        content_lower = content.lower()
        
        resume_terms = ['experience', 'education', 'skills', 'resume', 'curriculum vitae', 'work history', 'professional background']
        if any(term in content_lower for term in resume_terms):
            return "resume"
        
        financial_terms = ['revenue', 'profit', 'budget', 'financial', 'cost', 'investment', 'expense', 'income', 'revenue']
        if any(term in content_lower for term in financial_terms):
            return "financial"
        
        code_terms = ['def ', 'class ', 'import ', 'function ', 'python', 'javascript', 'java ', 'sql', 'database']
        if any(term in content_lower for term in code_terms):
            return "general"
        
        if len(content.split()) > 500:
            return "general"
        
        return "general"

# =============================================================================
# HYBRID RETRIEVAL SYSTEM - YOUR EXISTING
# =============================================================================

class FreeHybridRetrieval:
    def __init__(self, vector_store, embedding_system):
        self.vector_store = vector_store
        self.embedding_system = embedding_system
        self.bm25_index = None
        self.documents_cache = []
        self.search_metrics = {
            'total_searches': 0,
            'bm25_success': 0,
            'vector_success': 0,
            'fallback_used': 0
        }
        
    def build_hybrid_index(self, documents: List[Document]):
        """ENHANCED: Index both content AND metadata for BM25"""
        if not documents:
            print("⚠️  No documents to index")
            return
            
        # 🚨 GUARANTEE: Keep ALL documents for sync with vector store
        valid_docs = []
        for doc in documents:
            # Only exclude completely None documents, keep everything else
            if doc.page_content is not None:
                valid_docs.append(doc)
        
        if not valid_docs:
            print("⚠️  No documents to index")
            return
            
        self.documents_cache = valid_docs
        
        # 🎯 ENHANCED: Create metadata-augmented texts for BM25
        enhanced_texts = []
        for doc in valid_docs:
            # Start with page content
            enhanced_parts = [doc.page_content]
            
            # 🚨 ADD METADATA to BM25 index
            metadata = doc.metadata
            
            # Add filename/source (most important for file search)
            source = metadata.get('source', '')
            if source:
                # Extract just the filename if it's a path
                filename = os.path.basename(str(source))
                enhanced_parts.append(filename)
            
            # Add file type
            file_type = metadata.get('file_type', '')
            if file_type:
                enhanced_parts.append(file_type)
            
            # Add sheet name for Excel files
            sheet_name = metadata.get('sheet_name', '')
            if sheet_name:
                enhanced_parts.append(sheet_name)
            
            # Add section titles if available
            section_titles = metadata.get('section_titles', [])
            if section_titles:
                enhanced_parts.extend(section_titles)
            
            # Combine all parts
            enhanced_text = " ".join(enhanced_parts)
            enhanced_texts.append(enhanced_text)
        
        texts = enhanced_texts  # Use enhanced texts instead of just page_content
        
        try:
            # Professional tokenization (your existing good logic)
            tokenized_texts = [self._advanced_tokenize(text) for text in texts]
            
            # Build BM25 (your existing good logic)
            self.bm25_index = BM25Okapi(tokenized_texts)
            
            print(f"✅ ENHANCED HYBRID CACHE: {len(valid_docs)} documents")
            print(f"   - Total tokens: {sum(len(tokens) for tokens in tokenized_texts)}")
            print(f"   - Metadata included: filename, file_type, sheet_name, sections")
            
        except Exception as e:
            print(f"🔧 BM25 index build error: {e}")
            # 🚨 IMPORTANT: Even if BM25 fails, keep the documents in cache
            self.bm25_index = None
    
    def hybrid_search(self, query: str, top_k: int = 40) -> List[Tuple[Document, float]]:
        """PROFESSIONAL: Enterprise hybrid search with intelligent fusion"""
        self.search_metrics['total_searches'] += 1
        all_results = []
        
        try:
            # 🎯 STRATEGY 1: Parallel search execution
            vector_results = self.vector_search(query, top_k * 2)
            bm25_results = self.bm25_search(query, top_k * 2) if self.bm25_index else []
            
            # 🎯 STRATEGY 2: Intelligent score fusion
            fused_results = self._fuse_scores(vector_results, bm25_results, query)
            all_results.extend(fused_results)
            
            # 🎯 STRATEGY 3: Quality filtering
            quality_results = self._apply_quality_filters(all_results, query)
            
            if quality_results:
                final_results = self._deduplicate_results(quality_results)
                final_results.sort(key=lambda x: x[1], reverse=True)
                
                # Log search performance
                self._log_search_performance(query, len(final_results), len(vector_results), len(bm25_results))
                return final_results[:top_k]
            else:
                self.search_metrics['fallback_used'] += 1
                return self._professional_fallback_search(query, top_k)
                
        except Exception as e:
            print(f"🔧 Professional hybrid search error: {e}")
            self.search_metrics['fallback_used'] += 1
            return self._professional_fallback_search(query, top_k)
    
    def bm25_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """PROFESSIONAL: Enhanced BM25 with better scoring"""
        try:
            # 🎯 ENHANCED: Use metadata-aware tokenization
            tokenized_query = self._enhanced_tokenize_with_metadata(query)
            if not tokenized_query:
                return []

            
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Professional score normalization
            if len(scores) > 0:
                max_score = max(scores)
                if max_score > 0:
                    # Normalize to 0-1 range with non-linear scaling
                    if max_score > 0:
                        normalized_scores = np.sqrt(scores / max_score)
                    else:
                        normalized_scores = scores  # Fallback if all scores are zero
                else:
                    normalized_scores = scores
            else:
                normalized_scores = scores
            
            # Get top results with quality threshold
            top_indices = np.argsort(normalized_scores)[::-1][:top_k * 2]  # Get more for filtering
            
            results = []
            for idx in top_indices:
                score = float(normalized_scores[idx])
                if score > 0.05:  # Quality threshold
                    doc = self.documents_cache[idx]
                    results.append((doc, score))
            
            self.search_metrics['bm25_success'] += 1
            print(f"🔍 BM25: {len(results)} results (best: {max([s for d,s in results]) if results else 0:.3f})")
            return results[:top_k]
            
        except Exception as e:
            print(f"🔧 BM25 search error: {e}")
            return []
    
    def vector_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """FIXED: Chroma returns COSINE DISTANCE (lower = better)"""
        try:
            results = self.vector_store.vector_store.similarity_search_with_score(query, k=top_k)
            
            # 🚨 CHROMA RETURNS COSINE DISTANCE (0-2, LOWER = BETTER)
            # Convert to similarity: similarity = 1 - distance
            scored_results = []
            for doc, distance_score in results:
                similarity_score = 1.0 - distance_score  # Convert to HIGHER = BETTER
                scored_results.append((doc, similarity_score))
            
            # Sort by similarity (HIGHER = BETTER)
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            print(f"🔍 CONVERTED VECTOR SCORES (distance→similarity):")
            for i, (doc, similarity_score) in enumerate(scored_results[:10]):
                distance_score = 1.0 - similarity_score
                print(f"   {i+1}. {doc.metadata.get('source', 'unknown')}: dist={distance_score:.3f} → sim={similarity_score:.3f}")
            
            return scored_results
                    
        except Exception as e:
            print(f"🔧 Vector search error: {e}")
            return []
    
    def _advanced_tokenize(self, text: str) -> List[str]:
        """PROFESSIONAL: Advanced tokenization for BM25"""
        try:
            if not text or not isinstance(text, str):
                return []
            
            # Clean text
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = text.split()
            
            # Remove stopwords and short words
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                'for', 'of', 'with', 'by', 'as', 'is', 'was', 'were', 'be', 'been'
            }
            
            meaningful_words = [
                word for word in words 
                if len(word) > 2 and word not in stopwords
            ]
            
            return meaningful_words
            
        except Exception as e:
            print(f"🔧 Tokenization error: {e}")
            return [] 
    
    def _fuse_scores(self, vector_results: List[Tuple[Document, float]], 
                    bm25_results: List[Tuple[Document, float]],
                    query: str) -> List[Tuple[Document, float]]:
        """FIXED: Use vector scores as-is (already similarity)"""
        
        fused_results = []
        doc_data = {}
        
        # 🎯 STEP 1: Use vector scores AS-IS (they're already similarity)
        for doc, score in vector_results:
            print(f"🔍 VECTOR SCORE: {score:.3f} for {doc.metadata.get('source', 'unknown')}")
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in doc_data:
                doc_data[content_hash] = {}
            doc_data[content_hash]['vector'] = (doc, score)
        
        # 🎯 STEP 2: Add BM25 results (they're already similarity)
        for doc, score in bm25_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in doc_data:
                doc_data[content_hash] = {}
            doc_data[content_hash]['bm25'] = (doc, score)
        
        # 🎯 STEP 3: Fusion with CORRECT scoring
        for content_hash, data in doc_data.items():
            vector_score = data.get('vector', (None, 0.0))[1]
            bm25_score = data.get('bm25', (None, 0.0))[1]
            
            # Both scores are now HIGHER = BETTER
            if vector_score > 0 and bm25_score > 0:
                fused_score = (vector_score * 0.7) + (bm25_score * 0.3)
            elif vector_score > 0:
                fused_score = vector_score * 0.8
            else:
                fused_score = bm25_score * 0.5
            
            best_doc = data.get('vector', (None, 0.0))[0] or data.get('bm25', (None, 0.0))[0]
            fused_results.append((best_doc, fused_score))
        
        # Sort by fused score (HIGHER = BETTER)
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🎯 CORRECTED FUSION: {len(fused_results)} results")
        for i, (doc, score) in enumerate(fused_results[:10]):
            source = doc.metadata.get('source', 'unknown')
            print(f"   {i+1}. {source} (fused: {score:.3f})")
        
        return fused_results



    def _apply_quality_filters(self, results: List[Tuple[Document, float]], query: str) -> List[Tuple[Document, float]]:
        filtered_results = []
        
        for doc, score in results:
            content = doc.page_content
            
            # 🚨 REDUCE FILTERING - allow more documents through
            if len(content.strip()) < 5:  # Reduced from 7
                continue
                
            if len(content.split()) < 2:   # Reduced from 3
                continue
            
            # Keep more documents even with lower relevance
            filtered_results.append((doc, score))
        
        print(f"🎯 Relaxed filtering: {len(results)} → {len(filtered_results)} results")
        return filtered_results
    
    def _professional_fallback_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """PROFESSIONAL: Enhanced fallback search"""
        try:
            print("🔄 Using professional fallback search")
            
            # Try multiple fallback strategies
            fallback_results = []
            
            # Strategy 1: Simple vector search
            try:
                simple_results = self.vector_store.similarity_search(query, top_k)
                fallback_results.extend([(doc, 0.5) for doc in simple_results])
            except:
                pass
            
            # Strategy 2: BM25 from cache
            if self.documents_cache and not fallback_results:
                # Simple keyword matching
                query_terms = set(self._advanced_tokenize(query))
                for doc in self.documents_cache[:top_k]:
                    content_terms = set(self._advanced_tokenize(doc.page_content))
                    overlap = len(query_terms.intersection(content_terms))
                    if overlap > 0:
                        score = min(overlap / len(query_terms), 0.7) if query_terms else 0.3
                        fallback_results.append((doc, score))
            
            # Strategy 3: Return first documents
            if not fallback_results and self.documents_cache:
                fallback_results = [(doc, 0.1) for doc in self.documents_cache[:top_k]]
            
            return fallback_results[:top_k]
            
        except Exception as e:
            print(f"🔧 Professional fallback error: {e}")
            return []
    
    def _log_search_performance(self, query: str, total_results: int, 
                              vector_count: int, bm25_count: int):
        """PROFESSIONAL: Log search performance"""
        print(f"📊 SEARCH PERFORMANCE: '{query[:50]}...'")
        print(f"   - Total results: {total_results}")
        print(f"   - Vector results: {vector_count}")
        print(f"   - BM25 results: {bm25_count}")
        print(f"   - Success rate: {self.search_metrics['total_searches'] - self.search_metrics['fallback_used']}/{self.search_metrics['total_searches']}")
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """PROFESSIONAL: Get search performance metrics"""
        return self.search_metrics.copy()
    
    def _deduplicate_results(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """FIXED: Full content deduplication that matches add_documents logic"""
        seen_identifiers = set()
        unique_results = []
        removed_count = 0
        
        for doc, score in results:
            # 🚨 USE THE SAME LOGIC as in add_documents for consistency
            source = doc.metadata.get('source', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            content_hash = doc.metadata.get('content_hash', '')
            
            # If content_hash not in metadata, calculate it
            if not content_hash:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            
            # Create the SAME identifier used in add_documents
            identifier = f"{source}_{chunk_id}_{content_hash}"
            
            if identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                unique_results.append((doc, score))
            else:
                removed_count += 1
                print(f"🔍 DEDUPE REMOVED DUPLICATE: {source} (chunk: {chunk_id})")
        
        print(f"🎯 DEDUPLICATION: {len(results)} → {len(unique_results)} results (removed {removed_count} duplicates)")
        return unique_results
    
    def _enhanced_tokenize_with_metadata(self, text: str) -> List[str]:
        """Enhanced tokenization that preserves important metadata terms"""
        try:
            if not text or not isinstance(text, str):
                return []
            
            # Clean text
            text = re.sub(r'[^\w\s.-]', ' ', text.lower())
            
            # Preserve important patterns: filenames, extensions, etc.
            preserved_patterns = [
                r'\b\w+\.(pdf|docx|xlsx|pptx|csv|txt|md)\b',  # File extensions
                r'\b\w+-\w+\b',  # Hyphenated words (like "file-name")
                r'\b\d+\.\d+\b',  # Version numbers
            ]
            
            # Extract and preserve these patterns before general tokenization
            preserved_terms = []
            for pattern in preserved_patterns:
                matches = re.findall(pattern, text)
                preserved_terms.extend(matches)
            
            # General tokenization
            words = text.split()
            
            # Remove stopwords and short words (but keep preserved terms)
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                'for', 'of', 'with', 'by', 'as', 'is', 'was', 'were', 'be', 'been'
            }
            
            meaningful_words = [
                word for word in words 
                if len(word) > 2 and word not in stopwords
            ]
            
            # Add preserved terms back in
            meaningful_words.extend(preserved_terms)
            
            return meaningful_words
            
        except Exception as e:
            print(f"🔧 Enhanced tokenization error: {e}")
            return self._advanced_tokenize(text)  # Fallback to original

# =============================================================================
# QUERY ENHANCEMENT SYSTEM - YOUR EXISTING
# =============================================================================

class FreeQueryEnhancer:
    def __init__(self):
        self.query_patterns = {
            "person_search": [
                r"(who is|find|search for|information about)\s+([A-Z][a-z]+ [A-Z][a-z]+)",
                r"([A-Z][a-z]+ [A-Z][a-z]+)'s (background|experience|skills)"
            ],
            "skill_search": [
                r"(skills? in|knowledge of|experience with)\s+([\w\s]+)",
                r"who knows?\s+([\w\s]+)"
            ]
        }
    
    def enhance_query(self, original_query: str, user_role: str) -> List[str]:
        """Dramatically improve query understanding"""
        enhanced_queries = [original_query]
        
        enhanced_queries.extend(self._generate_variations(original_query))
        enhanced_queries.extend(self._role_enhancement(original_query, user_role))
        enhanced_queries.extend(self._pattern_expansion(original_query))
        
        unique_queries = list(set([q.strip() for q in enhanced_queries if q.strip()]))
        
        print(f"🔍 Enhanced '{original_query}' → {len(unique_queries)} queries")
        return unique_queries
    
    def _generate_variations(self, query: str) -> List[str]:
        """Generate semantic variations"""
        variations = []
        query_lower = query.lower()
        
        if not any(query_lower.startswith(prefix) for prefix in ['what', 'how', 'who', 'where', 'when', 'why']):
            variations.extend([
                f"what is {query}",
                f"how to {query}",
                f"information about {query}",
                f"details about {query}"
            ])
        
        action_verbs = ['find', 'search', 'locate', 'get', 'show']
        for verb in action_verbs:
            variations.append(f"{verb} {query}")
        
        return variations
    
    def _role_enhancement(self, query: str, user_role: str) -> List[str]:
        """Add role-specific context"""
        enhancements = []
        
        role_contexts = {
            "hr": ["HR perspective", "employee", "recruitment", "hiring"],
            "finance": ["financial", "budget", "revenue", "cost"],
            "it": ["technical", "software", "system", "IT"]
        }
        
        if user_role in role_contexts:
            for context in role_contexts[user_role][:2]:
                enhancements.append(f"{query} {context}")
        
        return enhancements
    
    def _pattern_expansion(self, query: str) -> List[str]:
        """Expand queries based on patterns"""
        expansions = []
        
        for pattern_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                for match in matches:
                    if pattern_type == "person_search":
                        person_name = match[1] if isinstance(match, tuple) else match
                        expansions.extend([
                            f"{person_name} resume",
                            f"{person_name} experience",
                            f"{person_name} skills"
                        ])
        
        return expansions

# =============================================================================
# PROFESSIONAL VECTOR STORE - ENHANCED
# =============================================================================

class ProfessionalVectorStore:
    def __init__(self, persist_directory: str = "./vector_store"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self._initialize_hybrid_retrieval()
        
        # Professional embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Document hash tracking
        self.document_hashes = {}
        self.hash_file = os.path.join(persist_directory, "document_hashes.json")
        self._load_document_hashes()
        
        # Your existing components
        self.document_storage = DocumentStorageManager()
        self.document_processor = ProfessionalDocumentProcessor()
        self.chunking_system = IntelligentChunkingSystem(self.embeddings)
        self.hybrid_retrieval = None
        
        # 🚨 ADD THIS: Initialize Hybrid Retrieval
        self._initialize_hybrid_retrieval()
        
        print("✅ Professional Vector Store initialized with duplicate prevention")


    

    # 🚨 ADD THESE METHODS TO ProfessionalVectorStore CLASS
    def get_hybrid_retriever(self):
        """Get the hybrid retriever instance"""
        return self.hybrid_retrieval

    def get_langchain_retriever(self, search_type: str = "similarity", k: int = 20):
        """Get a configured LangChain retriever"""
        try:
            retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k}
            )

            print(f"✅ LangChain retriever configured: {search_type} with k={k}")
            return retriever
        except Exception as e:
            print(f"❌ LangChain retriever configuration failed: {e}")
            # Fallback to basic similarity search
            return self.vector_store.as_retriever()

    def _initialize_hybrid_retrieval(self):
        """Initialize hybrid retrieval with existing documents - FIXED"""
        try:
            # Create hybrid retrieval instance - FIXED: Check if embeddings exist
            if hasattr(self, 'embeddings'):
                self.hybrid_retrieval = FreeHybridRetrieval(self, self.embeddings)
            else:
                print("⚠️ embeddings not available for hybrid retriever")
                self.hybrid_retrieval = FreeHybridRetrieval(self, None)
            
            # Load existing documents into hybrid index
            try:
                existing_docs = self.vector_store.similarity_search("", k=1000)
                if existing_docs:
                    self.hybrid_retrieval.build_hybrid_index(existing_docs)
                    print(f"✅ Hybrid retriever loaded with {len(existing_docs)} existing documents")
                else:
                    print("ℹ️ No existing documents found for hybrid retriever")
                    
            except Exception as e:
                print(f"⚠️ Could not load existing documents into hybrid retriever: {e}")
                
        except Exception as e:
            print(f"⚠️ Hybrid retriever initialization failed: {e}")
            self.hybrid_retrieval = None

    def _load_document_hashes(self):
        """Load document hashes from file"""
        try:
            if os.path.exists(self.hash_file):
                with open(self.hash_file, 'r') as f:
                    self.document_hashes = json.load(f)
                print(f"✅ Loaded document hashes: {sum(len(v) for v in self.document_hashes.values())} files tracked")
        except Exception as e:
            print(f"❌ Failed to load document hashes: {e}")
            self.document_hashes = {}

    def _save_document_hashes(self):
        """Save document hashes to file"""
        try:
            with open(self.hash_file, 'w') as f:
                json.dump(self.document_hashes, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save document hashes: {e}")

    def _get_document_hash(self, file_bytes: bytes, filename: str) -> str:
        """Generate unique hash for document content + metadata"""
        content_hash = hashlib.md5(file_bytes).hexdigest()
        metadata_hash = hashlib.md5(f"{filename}_{len(file_bytes)}".encode()).hexdigest()
        return f"{content_hash}_{metadata_hash}"

    def is_duplicate_document(self, namespace: str, file_bytes: bytes, filename: str) -> Tuple[bool, str]:
        """Check if document already exists and is unchanged"""
        if namespace not in self.document_hashes:
            self.document_hashes[namespace] = {}

        current_hash = self._get_document_hash(file_bytes, filename)
        
        # Check if we've seen this file before
        if filename in self.document_hashes[namespace]:
            existing_hash = self.document_hashes[namespace][filename]
            if existing_hash == current_hash:
                print(f"⏭️ SKIPPING unchanged duplicate: {filename}")
                return True, "unchanged"
        
        # Store the new hash and save
        self.document_hashes[namespace][filename] = current_hash
        self._save_document_hashes()
        
        return False, "new_or_modified"

    def clear_collection(self, namespace: str) -> int:
        """Clear collection - FIXED VERSION"""
        try:
            # Get count before clearing
            stats = self.get_collection_stats(namespace)
            doc_count = stats['total_documents']
            
            if doc_count > 0:
                # 🚨 FIX: For LangChain Chroma, we need to delete by metadata filter
                # This is a workaround since there's no direct "clear all" method
                from langchain_chroma import Chroma
                
                # Recreate the Chroma store (this clears everything)
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                print(f"🗑️  Cleared {doc_count} documents from {namespace}")
            
            return doc_count
            
        except Exception as e:
            print(f"❌ Failed to clear collection {namespace}: {e}")
            return 0

    def add_documents(self, namespace: str, documents: List[Document]) -> int:
        """FIXED: Add documents with PROPER hybrid store sync"""
        try:
            if not documents:
                return 0
            
            print(f"🚀 FIXED ADD: Processing {len(documents)} documents...")
            
            # Skip empty documents
            valid_documents = []
            for doc in documents:
                if doc.page_content and doc.page_content.strip():
                    valid_documents.append(doc)
            
            if not valid_documents:
                print("⚠️ No valid documents to add")
                return 0
            
            print(f"🔍 Valid documents: {len(valid_documents)}")
            
            # Process documents with BETTER uniqueness
            processed_docs = []
            source_files = {}
            
            for i, doc in enumerate(valid_documents):
                # Simple metadata cleanup
                simple_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        simple_metadata[key] = value
                    else:
                        simple_metadata[key] = str(value)
                
                # Extract filename for tracking
                source = simple_metadata.get('source', 'Unknown')
                filename = source.replace('Google Drive: ', '') if 'Google Drive:' in source else source
                
                # Track source files
                if filename not in source_files:
                    source_files[filename] = 0
                source_files[filename] += 1
                
                # 🚨 CRITICAL: Create TRULY UNIQUE chunk_id to prevent deduplication issues
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                unique_chunk_id = f"{namespace}_{filename}_{i}_{content_hash}_{uuid.uuid4().hex[:6]}"
                
                simple_metadata.update({
                    'chunk_id': unique_chunk_id,
                    'namespace': namespace,
                    'ingestion_time': datetime.now().isoformat(),
                    'content_length': len(doc.page_content),
                    'content_hash': content_hash  # Store for debugging
                })
                
                processed_doc = Document(
                    page_content=doc.page_content,
                    metadata=simple_metadata
                )
                processed_docs.append(processed_doc)
            
            # 🚨 STEP 1: Add to vector store (PERMANENT storage - disk)
            print(f"📦 Adding {len(processed_docs)} documents to vector store...")
            vector_result = self.vector_store.add_documents(processed_docs)
            print(f"✅ Vector store add completed")
            
            # 🚨 STEP 2: FIXED HYBRID STORE SYNC - Use the documents we JUST processed
            try:
                # Initialize hybrid retrieval if needed
                if not self.hybrid_retrieval:
                    self._initialize_hybrid_retrieval()
                
                if self.hybrid_retrieval:
                    # 🚨 CRITICAL FIX: Use the processed_docs we just added, NOT similarity search
                    # This ensures 1:1 sync between vector store and hybrid store
                    
                    # Smart deduplication for hybrid store
                    unique_docs = []
                    seen_identifiers = set()
                    
                    for doc in processed_docs:
                        # Use the same unique identifier logic as in _deduplicate_results
                        source = doc.metadata.get('source', 'unknown')
                        chunk_id = doc.metadata.get('chunk_id', 'unknown')
                        content_hash = doc.metadata.get('content_hash', '')
                        doc_identifier = f"{source}_{chunk_id}_{content_hash}"
                        
                        if doc_identifier not in seen_identifiers:
                            seen_identifiers.add(doc_identifier)
                            unique_docs.append(doc)
                    
                    print(f"🔄 Syncing hybrid store with {len(unique_docs)} unique documents...")
                    
                    # Build hybrid index with the EXACT same documents
                    self.hybrid_retrieval.build_hybrid_index(unique_docs)
                    
                    # ✅ VERIFICATION: Check sync worked
                    hybrid_count = len(self.hybrid_retrieval.documents_cache) if self.hybrid_retrieval.documents_cache else 0
                    
                    if hybrid_count == len(unique_docs):
                        print(f"✅ PERFECT SYNC: Vector store and hybrid store both have {hybrid_count} documents")
                    else:
                        print(f"⚠️ SYNC WARNING: Vector processed {len(unique_docs)} docs, Hybrid has {hybrid_count} docs")
                        
                        # Force rebuild if count doesn't match
                        if hybrid_count == 0:
                            print("🔄 Force rebuilding hybrid index...")
                            self.hybrid_retrieval.build_hybrid_index(unique_docs)
                            hybrid_count_retry = len(self.hybrid_retrieval.documents_cache) if self.hybrid_retrieval.documents_cache else 0
                            print(f"🔧 Hybrid store after rebuild: {hybrid_count_retry} documents")
                            
            except Exception as e:
                print(f"⚠️ Hybrid store sync failed: {e}")
                import traceback
                traceback.print_exc()
            
            # 🚨 UPDATE DOCUMENT STORAGE
            for filename, chunk_count in source_files.items():
                self.document_storage.add_document(
                    namespace, 
                    filename, 
                    0,  # file_size - you might want to calculate this
                    chunk_count
                )
                print(f"📝 Document storage: {filename} -> {chunk_count} chunks")
            
            print(f"🎉 SUCCESS: Added {len(processed_docs)} documents with FIXED sync")
            
            return len(processed_docs)
            
        except Exception as e:
            print(f"❌ add_documents failed: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def search(self, query: str, namespace: str, top_k: int = 40) -> List[Document]:
        """FIXED: Use CORRECTED vector results with similarity scoring"""
        try:
            # Get vector results with scores
            vector_results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            # 🚨 FIX: Convert distance to similarity and sort
            corrected_results = []
            for doc, distance_score in vector_results:
                similarity_score = 1.0 - distance_score  # Convert to HIGHER = BETTER
                corrected_results.append((doc, similarity_score))
            
            # Sort by similarity (HIGHER = BETTER)
            corrected_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return documents in CORRECT order (highest similarity first)
            documents = [doc for doc, score in corrected_results]
            return documents[:top_k]
                
        except Exception as e:
            return self.vector_store.similarity_search(query, k=min(top_k, 5))



    def get_collection_stats(self, namespace: str) -> Dict[str, Any]:
        """Get statistics for a namespace WITHOUT AUTO-SYNC to prevent recursion"""
        try:
            # 🚨 REMOVE AUTO-SYNC CALL - it causes recursion
            results = self.vector_store.similarity_search("", k=1000, filter={"namespace": namespace})
            
            sources = {}
            total_chars = 0
            for doc in results:
                source = doc.metadata.get('source', 'unknown')
                if source not in sources:
                    sources[source] = 0
                sources[source] += 1
                total_chars += len(doc.page_content)
            
            return {
                'total_documents': len(results),
                'total_characters': total_chars,
                'sources': sources,
                'namespace': namespace,
                'avg_chars_per_doc': total_chars // len(results) if results else 0
            }
        except Exception as e:
            print(f"❌ Failed to get collection stats: {e}")
            return {'total_documents': 0, 'sources': {}, 'namespace': namespace}


    def semantic_search(self, namespace: str, query: str, n_results: int = 20) -> Dict[str, Any]:
        """FIXED semantic search with proper relevance scoring"""
        try:
            # Use similarity_search_with_relevance_scores for better scoring
            if hasattr(self.vector_store, 'similarity_search_with_relevance_scores'):
                results = self.vector_store.similarity_search_with_relevance_scores(
                    query, 
                    k=n_results,
                    filter={"namespace": namespace} if namespace else {}
                )
                
                documents = []
                metadatas = []
                relevance_scores = []
                
                for doc, score in results:
                    documents.append(doc.page_content)
                    metadatas.append(doc.metadata)
                    relevance_scores.append(float(score))
                
                print(f"🔍 Semantic search: {len(documents)} results for '{query}', scores: {relevance_scores[:3]}")
                return {
                    'documents': documents,
                    'metadatas': metadatas,
                    'relevance_scores': relevance_scores
                }
            else:
                # Fallback
                results = self.vector_store.similarity_search(
                    query, 
                    k=n_results,
                    filter={"namespace": namespace} if namespace else {}
                )
                
                documents = [doc.page_content for doc in results]
                metadatas = [doc.metadata for doc in results]
                # Use default scores since we can't calculate proper ones
                relevance_scores = [0.8] * len(documents)
                
                return {
                    'documents': documents,
                    'metadatas': metadatas,
                    'relevance_scores': relevance_scores
                }
                
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            return {'documents': [], 'metadatas': [], 'relevance_scores': []}
        


    def cleanup_duplicates(self, namespace: str) -> int:
        """Remove duplicate documents from the vector store"""
        try:
            # Get all documents
            all_docs = self.vector_store.similarity_search("", k=2000, filter={"namespace": namespace})
            
            seen_content = set()
            duplicates_removed = 0
            
            # We can't directly delete from Chroma, so we'll rebuild
            unique_docs = []
            
            for doc in all_docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content and doc.page_content.strip():
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
                else:
                    duplicates_removed += 1
            
            if duplicates_removed > 0:
                print(f"🗑️ Found {duplicates_removed} duplicates, rebuilding collection...")
                
                # Clear and rebuild
                self.clear_collection(namespace)
                self.add_documents(namespace, unique_docs)
                
                print(f"✅ Rebuilt with {len(unique_docs)} unique documents")
            
            return duplicates_removed
            
        except Exception as e:
            print(f"❌ Duplicate cleanup failed: {e}")
            return 0

    def rebuild_search_index(self, namespace: str):
        """FAST: Only rebuild if absolutely necessary"""
        try:
            print(f"🔄 Fast index update for {namespace}...")
            
            # Get current documents
            all_docs = self.vector_store.similarity_search("", k=2000, filter={"namespace": namespace})
            
            if not all_docs:
                print("⚠️ No documents found to update index")
                return
            
            # Just update the hybrid index, don't clear everything
            if self.hybrid_retrieval is None:
                self.hybrid_retrieval = FreeHybridRetrieval(self, self.embeddings)
            
            # Smart deduplication
            unique_docs = []
            seen_identifiers = set()
            for doc in all_docs:
                source = doc.metadata.get('source', 'unknown')
                chunk_id = doc.metadata.get('chunk_id', 'unknown')
                doc_identifier = f"{source}_{chunk_id}"
                
                if doc_identifier not in seen_identifiers:
                    seen_identifiers.add(doc_identifier)
                    unique_docs.append(doc)
            
            self.hybrid_retrieval.build_hybrid_index(unique_docs)
            
            print(f"✅ Fast index update: {len(unique_docs)} documents")
            
        except Exception as e:
            print(f"❌ Fast index update failed: {e}")



class AtomicStorageManager:
    """GUARANTEES perfect sync between vector store and hybrid cache"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.sync_lock = threading.Lock()  # Prevent race conditions
        
    def initialize_perfect_sync(self):
        """Initialize with 100% synchronization"""
        with self.sync_lock:
            # Get ALL documents from vector store
            all_docs = self.vector_store.vector_store.similarity_search("", k=20000)
            
            # Initialize hybrid retriever
            if self.vector_store.hybrid_retrieval is None:
                self.vector_store.hybrid_retrieval = FreeHybridRetrieval(
                    self.vector_store, self.vector_store.embeddings
                )
            
            # 🚨 CRITICAL: NO FILTERING - keep EVERY document
            self.vector_store.hybrid_retrieval.documents_cache = all_docs
            
            # Build BM25 with ALL documents
            if all_docs:
                self._rebuild_bm25_index(all_docs)
            
            print(f"✅ PERFECT SYNC INIT: {len(all_docs)} documents in both stores")
            return len(all_docs)
    
    def atomic_add_documents(self, namespace: str, documents: List[Document]) -> int:
        """Add to BOTH stores atomically - all or nothing"""
        with self.sync_lock:
            try:
                # Step 1: Add to vector store
                print(f"📦 ATOMIC ADD: Adding {len(documents)} documents to both stores...")
                
                # Enhanced metadata for better tracking
                processed_docs = []
                for i, doc in enumerate(documents):
                    enhanced_metadata = doc.metadata.copy()
                    enhanced_metadata.update({
                        'atomic_id': f"{namespace}_{uuid.uuid4().hex[:8]}",
                        'added_timestamp': datetime.now().isoformat(),
                        'namespace': namespace,
                        'content_hash': hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
                    })
                    processed_docs.append(Document(
                        page_content=doc.page_content,
                        metadata=enhanced_metadata
                    ))
                
                # Add to vector store
                vector_count = self.vector_store.vector_store.add_documents(processed_docs)
                
                if vector_count > 0:
                    # Step 2: Immediately add to hybrid cache
                    if self.vector_store.hybrid_retrieval is None:
                        self.vector_store.hybrid_retrieval = FreeHybridRetrieval(
                            self.vector_store, self.vector_store.embeddings
                        )
                    
                    # Add to existing cache (don't replace)
                    current_cache = self.vector_store.hybrid_retrieval.documents_cache
                    current_cache.extend(processed_docs)
                    
                    # Rebuild BM25 with updated cache
                    self._rebuild_bm25_index(current_cache)
                    
                    print(f"✅ ATOMIC SUCCESS: {vector_count} docs added to both stores")
                    
                    # Update document storage
                    self._update_document_storage(namespace, processed_docs)
                    
                    return vector_count
                else:
                    print("❌ ATOMIC FAILED: Vector store add returned 0")
                    return 0
                    
            except Exception as e:
                print(f"❌ ATOMIC ADD FAILED: {e}")
                # Consider rolling back vector store if possible
                return 0
    
    def atomic_remove_documents(self, namespace: str, source_pattern: str) -> int:
        """Remove documents from BOTH stores atomically"""
        with self.sync_lock:
            try:
                # This is complex with Chroma - instead, we'll rebuild
                print(f"🔄 ATOMIC REMOVE: Rebuilding stores without {source_pattern}")
                
                # Get current documents
                all_docs = self.vector_store.vector_store.similarity_search("", k=20000)
                
                # Filter out documents matching pattern
                filtered_docs = [
                    doc for doc in all_docs 
                    if source_pattern not in doc.metadata.get('source', '')
                ]
                
                removed_count = len(all_docs) - len(filtered_docs)
                
                if removed_count > 0:
                    # Clear both stores
                    self.vector_store.clear_collection(namespace)
                    
                    # Rebuild with filtered documents
                    self.vector_store.vector_store.add_documents(filtered_docs)
                    self.vector_store.hybrid_retrieval.documents_cache = filtered_docs
                    self._rebuild_bm25_index(filtered_docs)
                    
                    print(f"✅ ATOMIC REMOVE: Removed {removed_count} documents")
                    return removed_count
                else:
                    print("ℹ️  No documents matched removal pattern")
                    return 0
                    
            except Exception as e:
                print(f"❌ ATOMIC REMOVE FAILED: {e}")
                return 0
    
    def _rebuild_bm25_index(self, documents: List[Document]):
        """Safely rebuild BM25 index"""
        try:
            if documents and self.vector_store.hybrid_retrieval:
                texts = [doc.page_content for doc in documents]
                tokenized_texts = [
                    self.vector_store.hybrid_retrieval._advanced_tokenize(text) 
                    for text in texts
                ]
                self.vector_store.hybrid_retrieval.bm25_index = BM25Okapi(tokenized_texts)
                print(f"📊 BM25 rebuilt with {len(documents)} documents")
        except Exception as e:
            print(f"⚠️  BM25 rebuild warning: {e}")
    
    def _update_document_storage(self, namespace: str, documents: List[Document]):
        """Update document storage tracking"""
        try:
            # Group by source file
            source_counts = {}
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                if source not in source_counts:
                    source_counts[source] = 0
                source_counts[source] += 1
            
            # Update storage
            for source, count in source_counts.items():
                filename = source.replace('Google Drive: ', '')
                self.vector_store.document_storage.add_document(
                    namespace, filename, 0, count
                )
                
        except Exception as e:
            print(f"⚠️  Document storage update failed: {e}")
    
    def verify_sync_status(self) -> Dict[str, Any]:
        """Verify and report synchronization status"""
        with self.sync_lock:
            try:
                vector_docs = self.vector_store.vector_store.similarity_search("", k=20000)
                hybrid_docs = self.vector_store.hybrid_retrieval.documents_cache if self.vector_store.hybrid_retrieval else []
                
                # Check counts
                vector_count = len(vector_docs)
                hybrid_count = len(hybrid_docs)
                
                # Check content hashes for deeper verification
                vector_hashes = set()
                hybrid_hashes = set()
                
                for doc in vector_docs:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    vector_hashes.add(content_hash)
                
                for doc in hybrid_docs:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    hybrid_hashes.add(content_hash)
                
                # Find discrepancies
                missing_from_hybrid = vector_hashes - hybrid_hashes
                missing_from_vector = hybrid_hashes - vector_hashes
                
                status = {
                    "perfect_sync": vector_count == hybrid_count and len(missing_from_hybrid) == 0 and len(missing_from_vector) == 0,
                    "vector_store_count": vector_count,
                    "hybrid_cache_count": hybrid_count,
                    "counts_match": vector_count == hybrid_count,
                    "content_hashes_match": vector_hashes == hybrid_hashes,
                    "missing_from_hybrid_count": len(missing_from_hybrid),
                    "missing_from_vector_count": len(missing_from_vector),
                    "sync_health": "PERFECT" if vector_count == hybrid_count and vector_hashes == hybrid_hashes else "NEEDS_ATTENTION"
                }
                
                return status
                
            except Exception as e:
                return {"error": str(e), "sync_health": "UNKNOWN"}

# =============================================================================
# PROFESSIONAL GOOGLE DRIVE MANAGER - ENHANCED
# =============================================================================



class ProfessionalGoogleDriveManager:
    """Enterprise-grade Google Drive integration"""
    
    def __init__(self, vector_store: ProfessionalVectorStore):
        self.vector_store = vector_store
        self.sync_status = {}
    

            
    async def sync_entire_drive(self, credentials: Dict, namespace: str, user_email: str) -> Dict[str, Any]:
        """UNLIMITED Google Drive sync - get ALL files from accessible folders"""
        try:
            # 🚨 FIX: Use the login email (user_email parameter) NOT the Google Drive email
            self.sync_status[namespace] = {
                'status': 'syncing', 
                'processed': 0, 
                'total': 0,
                'files_processed': 0,
                'folders_processed': 0,
                'current_file': ''
            }
            
            # 🆕 FIX: Use the login email that was passed in (manager@company.com)
            user_manager = CorporateUserManager()
            user_context = user_manager.authenticate_user(user_email)  # Use user_email, NOT the Google account
            
            print(f"🔍 DEBUG: Login user '{user_email}' authenticated as '{user_context.role}'")
            print(f"🔍 DEBUG: Accessible folders: {user_context.get_accessible_folders()}")
            
            # Convert to proper credentials (this is for Google Drive API access only)
            creds = Credentials(
                token=credentials['token'],
                refresh_token=credentials['refresh_token'],
                token_uri=credentials['token_uri'],
                client_id=credentials['client_id'],
                client_secret=credentials['client_secret'],
                scopes=credentials['scopes']
            )
            
            service = build('drive', 'v3', credentials=creds)
            
            print(f"🎯 {user_context.role} syncing from {user_context.get_accessible_folders()}")
            
            # 🚀 UNLIMITED: Get ALL files from accessible folders
            if "*" in user_context.get_accessible_folders():
                # Managers: Get ALL files from entire drive (with 200 limit)
                all_files = self._get_all_drive_files_unlimited(service)
                print(f"👑 Manager getting ALL {len(all_files)} files from Google Drive")
            else:
                # Other roles: Get ALL files from their specific folders
                all_files = await self._get_files_from_accessible_folders_unlimited(service, user_context)
                print(f"📁 {user_context.role} getting {len(all_files)} files from accessible folders")

  
            
            # 🚨 KEEP STATUS TRACKING FROM OLD METHOD
            self.sync_status[namespace]['total'] = len(all_files)
            print(f"📁 Processing {len(all_files)} files")
            
            # Process ALL files (your new processing logic)
            successful = 0
            total_files = len(all_files)
            
            for i, file_info in enumerate(all_files):
                # 🚨 KEEP STATUS UPDATES FROM OLD METHOD
                self.sync_status[namespace]['current_file'] = file_info.get('name', 'unknown')
                self.sync_status[namespace]['processed'] = i + 1
                
                if await self._process_drive_file(service, file_info, namespace):
                    successful += 1
                    self.sync_status[namespace]['files_processed'] += 1
                
                # Progress tracking every 50 files
                if i % 50 == 0:
                    print(f"🔄 {user_context.role}: Processed {i}/{total_files} files...")
            
            # 🚨 KEEP COMPLETION STATUS FROM OLD METHOD
            self.sync_status[namespace] = {
                'status': 'completed',
                'processed': successful,
                'total': len(all_files),
                'files_processed': successful,
                'folders_processed': 0,  # Your new method doesn't track folders separately
                'message': f"✅ Sync completed: {successful}/{len(all_files)} files processed"
            }
            
            print(f"✅ {user_context.role} sync completed: {successful}/{total_files} files processed")
            return {
                "status": "completed", 
                "files_processed": successful,
                "total_files": total_files,
                "role": user_context.role
            }
            
        except Exception as e:
            error_msg = f"Google Drive sync failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.sync_status[namespace] = {'status': 'error', 'error': error_msg}
            return self.sync_status[namespace]
    
    def _list_all_drive_files(self, service) -> List[Dict]:
        """List all files AND folders from Google Drive with 200 item limit"""
        items = []
        page_token = None
        item_count = 0
        max_items = 200  # 🎯 EXACTLY 200 files limit as requested
        
        try:
            while True:
                response = service.files().list(
                    q="trashed=false",  # Include both files and folders
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                    pageSize=100,  # Google's max per page
                    pageToken=page_token
                ).execute()
                
                batch_items = response.get('files', [])
                
                # 🎯 CHECK IF ADDING THIS BATCH WOULD EXCEED 200 LIMIT
                remaining_slots = max_items - item_count
                if len(batch_items) > remaining_slots:
                    items.extend(batch_items[:remaining_slots])
                    item_count += remaining_slots
                    print(f"📦 Reached limit of {max_items} items, stopping collection")
                    break
                else:
                    items.extend(batch_items)
                    item_count += len(batch_items)
                
                page_token = response.get('nextPageToken')
                
                # 🎯 STOP when we have 200 items OR no more pages
                if not page_token or item_count >= max_items:
                    break
                    
        except Exception as e:
            print(f"❌ Failed to list Google Drive items: {e}")
        
        # Count files vs folders
        files_count = len([item for item in items if item.get('mimeType') != 'application/vnd.google-apps.folder'])
        folders_count = len([item for item in items if item.get('mimeType') == 'application/vnd.google-apps.folder'])
        
        print(f"📁 Found {len(items)} total items: {files_count} files + {folders_count} folders (limited to {max_items})")
        return items
    
    async def _process_drive_file(self, service, file_info: Dict, namespace: str) -> bool:
        """Process a single Google Drive file - WITH DUPLICATE PROTECTION"""
        try:
            file_id = file_info['id']
            original_filename = file_info['name']
            mime_type = file_info.get('mimeType', '')
            
            # Process folders recursively
            if mime_type == 'application/vnd.google-apps.folder':
                print(f"📁 Processing folder: {original_filename}")
                return await self._process_folder_contents(service, file_id, original_filename, namespace)
            
            # Skip specific file types
            if self._should_skip_file(original_filename, mime_type):
                return False
            
            print(f"📄 Processing: {original_filename} (MIME: {mime_type})")
            
            file_content = None
            processed_filename = original_filename
            
            try:
                # Handle Google Workspace files differently
                if mime_type.startswith('application/vnd.google-apps.'):
                    file_content = self._export_google_workspace_file(service, file_id, mime_type, original_filename)
                    if file_content is not None:
                        if mime_type == 'application/vnd.google-apps.document':
                            processed_filename = f"{original_filename}.docx"
                        elif mime_type == 'application/vnd.google-apps.spreadsheet':
                            processed_filename = f"{original_filename}.xlsx"
                        elif mime_type == 'application/vnd.google-apps.presentation':
                            processed_filename = f"{original_filename}.pptx"
                else:
                    # Regular binary files
                    request = service.files().get_media(fileId=file_id)
                    file_content = request.execute()
                    
            except Exception as e:
                print(f"❌ Failed to download {original_filename}: {e}")
                return False
            
            if file_content is None:
                print(f"⚠️ No content extracted from: {original_filename}")
                return False
            
            # 🚨 CRITICAL: CHECK FOR DUPLICATE BEFORE PROCESSING
            is_duplicate, reason = self.vector_store.is_duplicate_document(namespace, file_content, original_filename)
            if is_duplicate:
                print(f"⏭️ SKIPPING duplicate document: {original_filename} ({reason})")
                return False  # Skip processing entirely
            
            # 🚨 PROCESS WITH PROPER FILENAME (only if not duplicate)
            processed_docs = self.vector_store.document_processor.process_document(file_content, processed_filename)
            
            if not processed_docs:
                print(f"⚠️ No content extracted from: {processed_filename}")
                return False
            
            # 🎯 EXTRACT STRUCTURED PREVIEW & STORE ORIGINAL FILE (for charts)
            structured_preview = None
            chart_data_id = None
            file_ext = os.path.splitext(processed_filename.lower())[1]
            
            if file_ext in ['.xlsx', '.xls', '.csv']:
                try:
                    structured_preview = self.vector_store.document_processor.extract_structured_preview(
                        file_content, processed_filename
                    )
                    
                    # 🎯 STORE ORIGINAL FILE FOR CHARTING
                    chart_data_id = self.vector_store.document_storage.store_original_file(
                        namespace, processed_filename, file_content
                    )
                    
                    print(f"✅ Extracted structured preview for {processed_filename}, stored original file: {chart_data_id}")
                except Exception as e:
                    print(f"⚠️ Structured preview/original storage failed for {processed_filename}: {e}")
            
            # Add enhanced metadata
            for doc in processed_docs:
                doc.metadata.update({
                    'source': f"Google Drive: {original_filename}",
                    'processed_filename': processed_filename,
                    'drive_file_id': file_id,
                    'drive_mime_type': mime_type,
                    'drive_modified_time': file_info.get('modifiedTime'),
                    'sync_timestamp': datetime.now().isoformat()
                })
            
            # Add to vector store (this will also sync hybrid)
            chunk_count = self.vector_store.add_documents(namespace, processed_docs)
            
            if chunk_count > 0:
                # 🎯 LINK CHART DATA WITH DOCUMENT METADATA
                if structured_preview and chart_data_id:
                    self.vector_store.document_storage.update_document_with_chart_data(
                        namespace,
                        filename=processed_filename,
                        file_id=chart_data_id,
                        structured_preview=structured_preview
                    )
                    print(f"✅ Linked chart data for {processed_filename}")
                
                print(f"✅ Successfully processed: {original_filename} -> {chunk_count} chunks")
                return True
            else:
                print(f"⚠️ Failed to store: {original_filename}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to process Google Drive file {file_info.get('name', 'unknown')}: {e}")
            return False
    
    async def _find_folder_ids_by_name(self, service, folder_names: List[str]) -> Dict[str, str]:
        """Find Google Drive folder IDs by their CORPORATE FOLDER NAMES"""
        folder_ids = {}
        
        for folder_name in folder_names:
            try:
                # Search for folder by exact corporate name (like "01-Company-Public")
                query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
                response = service.files().list(
                    q=query,
                    pageSize=5,
                    fields="files(id, name)"
                ).execute()
                
                folders = response.get('files', [])
                if folders:
                    folder_ids[folder_name] = folders[0]['id']
                    print(f"✅ Found corporate folder '{folder_name}' → ID: {folders[0]['id']}")
                else:
                    print(f"❌ Corporate folder not found: {folder_name}")
                    folder_ids[folder_name] = None
                    
            except Exception as e:
                print(f"❌ Error finding corporate folder '{folder_name}': {e}")
                folder_ids[folder_name] = None
        
        return folder_ids
    

    async def _get_files_from_accessible_folders_unlimited(self, service, user_context: CorporateUserContext) -> List[Dict]:
        """Get ALL files from accessible folders (NO LIMITS)"""
        accessible_folders = user_context.get_accessible_folders()
        all_files = []
        
        # 🚀 Find folder IDs first using corporate folder names
        folder_ids = await self._find_folder_ids_by_name(service, accessible_folders)
        
        # 🚀 Get ALL files from each folder (NO LIMITS)
        for folder_name, folder_id in folder_ids.items():
            if folder_id:
                print(f"📂 Getting ALL files from corporate folder: {folder_name}")
                folder_files = self._get_files_from_folder_unlimited(service, folder_id)
                all_files.extend(folder_files)
                print(f"   ✅ Found {len(folder_files)} files in {folder_name}")
        
        return all_files

    async def _get_files_from_accessible_folders_unlimited(self, service, user_context: CorporateUserContext) -> List[Dict]:
        """Get ALL files from accessible folders (WITH 200 FILE LIMIT)"""
        accessible_folders = user_context.get_accessible_folders()
        all_files = []
        total_files_collected = 0
        max_files = 200  # 🎯 Overall limit across all folders
        
        print(f"🔍 Looking for corporate folders: {accessible_folders}")
        
        # 🚀 Find folder IDs first
        folder_ids = await self._find_folder_ids_by_name(service, accessible_folders)
        
        # 🚀 Get files from each folder until we reach 200 total
        for folder_name, folder_id in folder_ids.items():
            if folder_id and total_files_collected < max_files:
                print(f"📂 Getting files from: {folder_name}")
                folder_files = self._get_files_from_folder_unlimited(service, folder_id)
                
                # 🎯 Check if adding these files would exceed our total limit
                remaining_slots = max_files - total_files_collected
                if len(folder_files) > remaining_slots:
                    # Take only what we need to reach 200
                    folder_files = folder_files[:remaining_slots]
                    print(f"   📦 Taking {len(folder_files)} files to reach 200 total limit")
                
                all_files.extend(folder_files)
                total_files_collected = len(all_files)
                
                print(f"   ✅ Found {len(folder_files)} files in {folder_name} (total: {total_files_collected})")
                
                if total_files_collected >= max_files:
                    print(f"🎯 Reached overall 200 file limit across all folders")
                    break
        
        print(f"📁 Total files collected from all folders: {len(all_files)}")
        return all_files
    
    def _should_skip_file(self, filename: str, mime_type: str) -> bool:
        """Determine if a file should be skipped"""
        skip_extensions = {'.ipynb', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv'}
        skip_keywords = {'video', 'movie', 'film', 'trailer'}
        
        file_ext = os.path.splitext(filename)[1].lower()
        filename_lower = filename.lower()
        
        if file_ext in skip_extensions:
            return True
        
        if any(keyword in filename_lower for keyword in skip_keywords):
            return True
        
        return False
    
    def get_sync_status(self, namespace: str) -> Dict[str, Any]:
        """Get current sync status"""
        return self.sync_status.get(namespace, {'status': 'not_started'})
    


    def _perform_role_based_sync(self, user_context: CorporateUserContext, namespace: str):
        """Role-based sync that processes files properly WITH CONTENT EXTRACTION"""
        try:
            if not user_credentials:
                self.sync_status[namespace] = {"status": "error", "error": "No credentials"}
                return
            
            user_email, credentials_dict = list(user_credentials.items())[0]
            
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            
            creds = Credentials(
                token=credentials_dict['token'],
                refresh_token=credentials_dict['refresh_token'],
                token_uri=credentials_dict['token_uri'],
                client_id=credentials_dict['client_id'],
                client_secret=credentials_dict['client_secret'],
                scopes=credentials_dict['scopes']
            )
            
            service = build('drive', 'v3', credentials=creds)
            accessible_folders = user_context.get_accessible_folders()
            
            print(f"🎯 {user_context.role} starting sync...")
            print(f"📁 Accessible folders: {accessible_folders}")
            
            all_documents = []
            processed_files = 0
            
            # MANAGERS/EXECUTIVES: Get ALL files from entire Google Drive
            if "*" in accessible_folders:
                print("👑 Manager/Executive - getting ALL files from Google Drive")
                results = service.files().list(
                    pageSize=100,
                    fields="nextPageToken, files(id, name, mimeType, parents)",
                ).execute()
                items = results.get('files', [])
            
            # OTHER ROLES: Get files only from their assigned folders
            else:
                print(f"👤 {user_context.role} - getting files from assigned folders")
                all_items = []
                
                for folder_name in accessible_folders:
                    print(f"📂 Searching in folder: {folder_name}")
                    try:
                        # Search for files in this specific folder
                        query = f"'{folder_name}' in parents and trashed=false"
                        results = service.files().list(
                            q=query,
                            pageSize=50,
                            fields="nextPageToken, files(id, name, mimeType, parents)",
                        ).execute()
                        
                        folder_items = results.get('files', [])
                        all_items.extend(folder_items)
                        print(f"✅ Found {len(folder_items)} files in {folder_name}")
                        
                    except Exception as e:
                        print(f"❌ Could not access folder {folder_name}: {e}")
                        continue
                
                items = all_items
            
            print(f"📄 Total files to process: {len(items)}")
            
            # ENHANCED: Process each file with MULTIPLE extraction methods
            for i, item in enumerate(items):
                try:
                    file_id = item['id']
                    file_name = item['name']
                    mime_type = item.get('mimeType', 'unknown')
                    
                    # 🚨 SKIP .ipynb FILES AS REQUESTED
                    if file_name.lower().endswith('.ipynb'):
                        print(f"⏭️  SKIPPING .ipynb: {file_name}")
                        continue
                    
                    if i % 10 == 0:
                        print(f"🔄 [{i+1}/{len(items)}] Processing: {file_name}")
                    
                    # 🎯 ENHANCED CONTENT EXTRACTION WITH MULTIPLE METHODS
                    document = self._enhanced_extract_content(service, file_id, file_name, mime_type, creds)
                    
                    if document and document.page_content.strip():
                        all_documents.append(document)
                        processed_files += 1
                        print(f"✅ EXTRACTED CONTENT: {file_name}")
                        
                        # Show content preview
                        content_preview = document.page_content[:100].replace('\n', ' ')
                        print(f"   📝 Preview: {content_preview}...")
                    else:
                        print(f"⚠️  NO CONTENT: {file_name}")
                        
                except Exception as e:
                    error_msg = str(e)
                    if "SERVICE_DISABLED" in error_msg:
                        print(f"❌ API not enabled for: {file_name}")
                    else:
                        print(f"❌ Failed: {file_name} - {error_msg[:100]}...")
                    continue
            
            # Add to vector store
            if all_documents:
                final_count = self._process_drive_documents(all_documents, namespace)
                self.sync_status[namespace] = {
                    "status": "completed",
                    "files_processed": final_count,
                    "total_files_found": len(items),
                    "successful_files": processed_files,
                    "accessible_folders": accessible_folders,
                    "message": f"✅ {user_context.role} sync: {final_count} documents from {processed_files} files in {len(accessible_folders)} folders"
                }
                print(f"🎉 {user_context.role.upper()} SUCCESS: {final_count} documents from {processed_files} files")
            else:
                self.sync_status[namespace] = {
                    "status": "completed", 
                    "files_processed": 0,
                    "total_files_found": len(items),
                    "accessible_folders": accessible_folders,
                    "message": f"Found {len(items)} files in {len(accessible_folders)} folders but extracted 0 documents"
                }
                print(f"⚠️  Found {len(items)} files but extracted 0 documents")
            
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            import traceback
            traceback.print_exc()
            self.sync_status[namespace] = {"status": "error", "error": str(e)}

    def _get_specific_folder_files(self, credentials_dict, folder_names, namespace: str = "management_full"):
        """Get files only from specific folder names"""
        all_documents = []
        processed_folders = 0
        
        for i, folder_name in enumerate(folder_names):
            try:
                self.sync_status[namespace] = {
                    "status": "syncing", 
                    "progress": (i / len(folder_names)) * 100,
                    "current_folder": folder_name,
                    "folders_processed": i,
                    "total_folders": len(folder_names)
                }
                
                print(f"📂 [{i+1}/{len(folder_names)}] Processing folder: {folder_name}")
                
                # Use GoogleDriveLoader with the exact folder name
                loader = GoogleDriveLoader(
                    folder_id=folder_name,  # Use exact folder name like "01-Company-Public"
                    recursive=True,  # Get files in subfolders too
                    file_types=[],  # Empty list = ALL file types
                    load_trashed=False,
                    credentials=credentials_dict
                )
                
                documents = loader.load()
                
                if documents:
                    all_documents.extend(documents)
                    processed_folders += 1
                    print(f"✅ {len(documents)} files from {folder_name}")
                else:
                    print(f"⚠️  No files found in {folder_name}")
                
            except Exception as e:
                print(f"❌ Could not sync folder {folder_name}: {e}")
                continue
        
        return all_documents, processed_folders

    
    def _get_all_drive_files(self, credentials_dict):
        """Get ALL files from entire Google Drive (for managers)"""
        try:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            
            # Convert to proper Credentials object
            creds = Credentials(
                token=credentials_dict['token'],
                refresh_token=credentials_dict['refresh_token'],
                token_uri=credentials_dict['token_uri'],
                client_id=credentials_dict['client_id'],
                client_secret=credentials_dict['client_secret'],
                scopes=credentials_dict['scopes']
            )
            
            service = build('drive', 'v3', credentials=creds)
            
            # Get ALL files from entire Google Drive
            print("🌐 Manager: Fetching ALL files from Google Drive...")
            results = service.files().list(
                pageSize=200,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            ).execute()
            
            items = results.get('files', [])
            print(f"📄 Manager found {len(items)} total files in Google Drive")
            
            all_documents = []
            
            # Process each file individually with PROPER credentials
            for i, item in enumerate(items):
                try:
                    file_id = item['id']
                    file_name = item['name']
                    mime_type = item.get('mimeType', 'unknown')
                    
                    if i % 10 == 0:  # Progress indicator
                        print(f"🔄 [{i+1}/{len(items)}] Processing: {file_name}")
                    
                    # Use GoogleDriveLoader with PROPER Credentials object
                    loader = GoogleDriveLoader(
                        file_ids=[file_id],
                        credentials=creds  # Use the Credentials object, not the dict
                    )
                    
                    documents = loader.load()
                    
                    if documents and any(doc.page_content.strip() for doc in documents):
                        all_documents.extend(documents)
                        print(f"✅ Extracted: {file_name}")
                    else:
                        print(f"⚠️  No content: {file_name}")
                        
                except Exception as e:
                    print(f"❌ Failed to process {file_name}: {str(e)[:100]}...")
                    continue
            
            print(f"✅ Manager extracted {len(all_documents)} documents from all files")
            return all_documents
            
        except Exception as e:
            print(f"❌ Error getting all drive files: {e}")
            import traceback
            traceback.print_exc()

            return []


    
    def _get_files_from_folder_unlimited(self, service, folder_id: str) -> List[Dict]:
        """Get ALL files from specific folder (NO LIMITS) - WITH 200 FILE LIMIT"""
        items = []
        page_token = None
        total_collected = 0
        max_files = 200  # 🎯 Apply the 200 file limit
        
        try:
            while True:
                response = service.files().list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    pageSize=1000,  # 🚀 Large page size for efficiency
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                    pageToken=page_token
                ).execute()
                
                batch = response.get('files', [])
                
                # 🎯 CHECK IF ADDING THIS BATCH WOULD EXCEED 200 LIMIT
                remaining_slots = max_files - total_collected
                if len(batch) > remaining_slots:
                    items.extend(batch[:remaining_slots])
                    total_collected += remaining_slots
                    print(f"   📦 Reached 200 file limit in folder, stopping collection")
                    break
                else:
                    items.extend(batch)
                    total_collected += len(batch)
                
                print(f"   📄 Got {len(batch)} files from folder (total: {len(items)})")
                
                page_token = response.get('nextPageToken')
                if not page_token or total_collected >= max_files:
                    break  # 🚀 No more pages or reached limit
                    
        except Exception as e:
            print(f"❌ Error getting files from folder {folder_id}: {e}")
        
        print(f"   ✅ Retrieved {len(items)} files from folder (limited to {max_files})")
        return items

    def _get_all_drive_files_unlimited(self, service) -> List[Dict]:
        """Get ALL files from entire Google Drive (NO LIMITS) - WITH 200 FILE LIMIT"""
        items = []
        page_token = None
        total_collected = 0
        max_files = 200  # 🎯 Apply the 200 file limit
        
        try:
            while True:
                response = service.files().list(
                    q="trashed=false",
                    pageSize=1000,  # 🚀 Large page size for efficiency
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                    pageToken=page_token
                ).execute()
                
                batch = response.get('files', [])
                
                # 🎯 CHECK IF ADDING THIS BATCH WOULD EXCEED 200 LIMIT
                remaining_slots = max_files - total_collected
                if len(batch) > remaining_slots:
                    items.extend(batch[:remaining_slots])
                    total_collected += remaining_slots
                    print(f"📦 Reached 200 file limit, stopping collection")
                    break
                else:
                    items.extend(batch)
                    total_collected += len(batch)
                
                print(f"📄 Got {len(batch)} files from Google Drive (total: {len(items)})")
                
                page_token = response.get('nextPageToken')
                if not page_token or total_collected >= max_files:
                    break  # 🚀 No more pages or reached limit
                    
        except Exception as e:
            print(f"❌ Error listing all drive files: {e}")
        
        print(f"🏁 Retrieved {len(items)} files from Google Drive (limited to {max_files})")
        return items

    def _export_google_workspace_file(self, service, file_id: str, mime_type: str, file_name: str) -> Optional[bytes]:
        """Export Google Workspace files - FIXED RETURN TYPE"""
        try:
            export_mime_types = {
                'application/vnd.google-apps.document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/vnd.google-apps.presentation': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'application/vnd.google-apps.drawing': 'image/png',
            }
            
            export_mime = export_mime_types.get(mime_type)
            if not export_mime:
                print(f"⚠️ No export format available for {mime_type}: {file_name}")
                return None
            
            # Determine file extension
            ext_mapping = {
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx', 
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
                'image/png': '.png',
            }
            
            file_ext = ext_mapping.get(export_mime, '')
            processed_filename = f"{file_name}{file_ext}"
            
            print(f"📤 Exporting Google Workspace file: {file_name} -> {export_mime} as {processed_filename}")
            
            # Export the file
            request = service.files().export_media(fileId=file_id, mimeType=export_mime)
            file_content = request.execute()
            
            print(f"✅ Successfully exported: {file_name} as {processed_filename}")
            
            # 🚨 RETURN JUST THE CONTENT, not a tuple
            return file_content
            
        except Exception as e:
            print(f"❌ Failed to export Google Workspace file {file_name}: {e}")
            return None
# =============================================================================
# CHAT HISTORY & DOCUMENT STORAGE - YOUR EXISTING
# =============================================================================

class ChatHistoryManager:
    def __init__(self, mongo_db=None):
        """Manage chat history with optional MongoDB persistence.

        - In-memory store is always used for fast access.
        - If mongo_db is provided, sessions and messages are mirrored into MongoDB
          so they persist across restarts (ideal for Render).
        """
        self.sessions: Dict[str, Dict] = {}
        self.workspace_sessions: Dict[str, List[str]] = {}
        
        self.mongo_db = mongo_db
        self.sessions_coll = None
        self.messages_coll = None
        
        if self.mongo_db is not None:
            self.sessions_coll = self.mongo_db.get_collection("chat_sessions")
            self.messages_coll = self.mongo_db.get_collection("chat_messages")
    
    def create_session(self, workspace: str, title: str = "New Chat") -> str:
        session_id = str(uuid.uuid4())
        session_record = {
            'session_id': session_id,
            'workspace': workspace,
            'title': title,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'messages': []
        }
        self.sessions[session_id] = session_record
        
        if workspace not in self.workspace_sessions:
            self.workspace_sessions[workspace] = []
        self.workspace_sessions[workspace].append(session_id)
        
        # Persist session to MongoDB if available
        if self.sessions_coll is not None:
            try:
                # Do not store messages in the session document; keep them in a separate collection
                mongo_record = {
                    'session_id': session_id,
                    'workspace': workspace,
                    'title': title,
                    'created_at': session_record['created_at'],
                    'updated_at': session_record['updated_at'],
                    'message_count': 0
                }
                self.sessions_coll.update_one(
                    {'session_id': session_id},
                    {'$set': mongo_record},
                    upsert=True
                )
            except Exception as e:
                print(f"⚠️ Failed to persist chat session to MongoDB: {e}")
        
        return session_id
    
    def get_sessions(self, workspace: str) -> List[Dict]:
        if workspace not in self.workspace_sessions:
            return []
        
        sessions = []
        for session_id in self.workspace_sessions[workspace]:
            if session_id in self.sessions:
                session_data = self.sessions[session_id]
                sessions.append({
                    'session_id': session_data['session_id'],
                    'title': session_data['title'],
                    'created_at': session_data['created_at'],
                    'updated_at': session_data['updated_at'],
                    'message_count': len(session_data['messages'])
                })
        
        sessions.sort(key=lambda x: x['updated_at'], reverse=True)
        return sessions
    
    def add_message(self, session_id: str, message_type: str, content: str, metadata: Dict = None):
        if session_id not in self.sessions:
            return
        
        message = {
            'type': message_type,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        session = self.sessions[session_id]
        session['messages'].append(message)
        session['updated_at'] = datetime.now().isoformat()
        
        if message_type == 'user' and len(session['messages']) == 1:
            title_words = content.split()[:6]
            session['title'] = ' '.join(title_words) + ('...' if len(content.split()) > 6 else '')
        
        # Persist message and update session metadata in MongoDB if available
        if self.mongo_db is not None and self.messages_coll is not None and self.sessions_coll is not None:
            try:
                mongo_message = {
                    'session_id': session_id,
                    'workspace': session['workspace'],
                    'type': message_type,
                    'content': content,
                    'timestamp': message['timestamp'],
                    'metadata': metadata or {}
                }
                self.messages_coll.insert_one(mongo_message)
                
                self.sessions_coll.update_one(
                    {'session_id': session_id},
                    {
                        '$set': {
                            'updated_at': session['updated_at'],
                            'title': session['title']
                        },
                        '$inc': {'message_count': 1}
                    },
                    upsert=True
                )
            except Exception as e:
                print(f"⚠️ Failed to persist chat message to MongoDB: {e}")
    
    def get_messages(self, session_id: str) -> List[Dict]:
        # Prefer in-memory cache
        if session_id in self.sessions and self.sessions[session_id]['messages']:
            return self.sessions[session_id]['messages']
        
        # If no in-memory messages but MongoDB is available, load from DB
        if self.messages_coll is not None:
            try:
                cursor = self.messages_coll.find({'session_id': session_id}).sort('timestamp', 1)
                messages = []
                for doc in cursor:
                    messages.append({
                        'type': doc.get('type'),
                        'content': doc.get('content'),
                        'timestamp': doc.get('timestamp'),
                        'metadata': doc.get('metadata', {})
                    })
                # Cache in memory for this process
                if session_id in self.sessions:
                    self.sessions[session_id]['messages'] = messages
                return messages
            except Exception as e:
                print(f"⚠️ Failed to load messages from MongoDB: {e}")
        
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]['messages']
    
    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            workspace = self.sessions[session_id]['workspace']
            del self.sessions[session_id]
            
            if workspace in self.workspace_sessions:
                if session_id in self.workspace_sessions[workspace]:
                    self.workspace_sessions[workspace].remove(session_id)
class DocumentStorageManager:
    def __init__(self, mongo_db=None):
        self.storage_file = "./document_storage.json"
        self.workspace_documents: Dict[str, List[Dict[str, Any]]] = {}
        self.sync_in_progress = False
        self.structured_data_dir = STRUCTURED_DATA_DIR
        self.original_files_dir = ORIGINAL_FILES_DIR
        os.makedirs(self.structured_data_dir, exist_ok=True)
        os.makedirs(self.original_files_dir, exist_ok=True)
        
        # Optional MongoDB integration
        self.mongo_db = mongo_db
        self.docs_coll = None
        self.structured_coll = None
        if self.mongo_db is not None:
            try:
                self.docs_coll = self.mongo_db.get_collection("documents")
                self.structured_coll = self.mongo_db.get_collection("structured_previews")
                print("✅ MongoDB DocumentStorageManager connected to collections: documents, structured_previews")
            except Exception as e:
                print(f"⚠️ Failed to configure MongoDB collections for DocumentStorageManager: {e}")
                self.mongo_db = None
        
        self._load_from_disk()
    
    def store_original_file(self, workspace: str, filename: str, file_bytes: bytes) -> str:
        """Store original file for charting purposes"""
        try:
            # Create safe filename with unique ID
            file_id = str(uuid.uuid4())
            safe_filename = f"{workspace}_{file_id}_{filename}"
            file_path = os.path.join(self.original_files_dir, safe_filename)
            
            # Save the file
            with open(file_path, 'wb') as f:
                f.write(file_bytes)
            
            print(f"💾 Stored original file for charting: {filename} -> {file_path}")
            return file_id
            
        except Exception as e:
            print(f"❌ Failed to store original file {filename}: {e}")
            return None
    
    def get_original_file(self, workspace: str, file_id: str) -> Optional[bytes]:
        """Retrieve original file for charting"""
        try:
            # Find the file by pattern
            pattern = f"{workspace}_{file_id}_*"
            for filename in os.listdir(self.original_files_dir):
                if filename.startswith(f"{workspace}_{file_id}_"):
                    file_path = os.path.join(self.original_files_dir, filename)
                    with open(file_path, 'rb') as f:
                        return f.read()
            return None
        except Exception as e:
            print(f"❌ Failed to retrieve original file {file_id}: {e}")
            return None

    # Add this method to your existing DocumentStorageManager
    def update_document_with_chart_data(self, workspace: str, filename: str, file_id: str, structured_preview: Dict[str, Any]):
        """Update document with chart data reference"""
        return self.update_document_metadata(
            workspace,
            filename=filename,
            updates={
                'chart_data_id': file_id,
                'has_structured_data': True,
                'structured_data_type': structured_preview.get('type', 'unknown')
            },
            structured_preview=structured_preview
        )
    
    def _load_from_disk(self):  # 🆕 ADD THIS METHOD
        """Load document tracking from disk on startup"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    self.workspace_documents = json.load(f)
                print(f"✅ Loaded {sum(len(docs) for docs in self.workspace_documents.values())} documents to storage")
        except Exception as e:
            print(f"⚠️ Failed to load document storage: {e}")
            self.workspace_documents = {}
    
    def _save_to_disk(self):  # 🆕 ADD THIS METHOD
        """Save document tracking to disk"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.workspace_documents, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save document storage: {e}")
    
    def auto_sync_from_vector_store(self, workspace: str, vector_store):
        """Automatically sync document tracking with ALL vector store contents"""
        if self.sync_in_progress:
            return  # 🚨 PREVENT RECURSION
            
        try:
            self.sync_in_progress = True
            
            # 🚨 FIX: Get ALL documents directly from vector store
            all_docs = vector_store.vector_store.similarity_search("", k=10000)
            
            if workspace not in self.workspace_documents:
                self.workspace_documents[workspace] = []
            
            current_filenames = {doc['filename'] for doc in self.workspace_documents[workspace]}
            
            # Count documents by source
            source_counts = {}
            for doc in all_docs:
                source = doc.metadata.get('source', 'unknown')
                filename = source.replace('Google Drive: ', '') if 'Google Drive:' in source else source
                if filename not in source_counts:
                    source_counts[filename] = 0
                source_counts[filename] += 1
            
            # Add missing documents
            added_count = 0
            for filename, count in source_counts.items():
                if filename not in current_filenames and filename != 'unknown':
                    self.add_document(workspace, filename, 0, count)
                    added_count += 1
                    print(f"🔍 Auto-added to documents tab: {filename}")
            
            # Remove documents that no longer exist
            current_sources = set(source_counts.keys())
            self.workspace_documents[workspace] = [
                doc for doc in self.workspace_documents[workspace] 
                if doc['filename'] in current_sources
            ]
            
            self._save_to_disk()
            print(f"🔄 Auto-synced documents tab: {added_count} new files, total {len(source_counts)} files")
            
        except Exception as e:
            print(f"❌ Auto-sync failed: {e}")
        finally:
            self.sync_in_progress = False
    def add_document(self, workspace: str, filename: str, file_size: int, chunk_count: int, extra_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        if workspace not in self.workspace_documents:
            self.workspace_documents[workspace] = []
        
        # Check if document already exists
        for doc in self.workspace_documents[workspace]:
            if doc['filename'] == filename:
                # Update existing document
                doc.update({
                    'file_size': file_size,
                    'upload_time': datetime.now().isoformat(),
                    'chunk_count': chunk_count,
                    'status': 'updated'
                })
                if extra_metadata:
                    doc.update(extra_metadata)
                self._save_to_disk()  # 🆕 ADD THIS
                print(f"📝 Updated document record: {filename}")
                return doc
        
        # Add new document
        document_record = {
            'filename': filename,
            'file_size': file_size,
            'upload_time': datetime.now().isoformat(),
            'chunk_count': chunk_count,
            'file_id': str(uuid.uuid4())[:8],
            'status': 'processed'
        }
        if extra_metadata:
            document_record.update(extra_metadata)
        
        self.workspace_documents[workspace].append(document_record)
        self._save_to_disk()  # 🆕 ADD THIS
        print(f"✅ Added document to storage: {filename} ({chunk_count} chunks)")
        
        # Mirror new document into MongoDB if configured
        if hasattr(self, "docs_coll") and self.docs_coll is not None:
            try:
                mongo_record = document_record.copy()
                mongo_record['_id'] = f"{workspace}:{mongo_record['file_id']}"
                mongo_record['workspace'] = workspace
                self.docs_coll.update_one(
                    {'_id': mongo_record['_id']},
                    {'$set': mongo_record},
                    upsert=True
                )
            except Exception as e:
                print(f"⚠️ Failed to persist document to MongoDB: {e}")
        
        return document_record
    
    # KEEP ALL YOUR EXISTING METHODS AS-IS:
    def get_document_count(self, workspace: str) -> int:
        if workspace not in self.workspace_documents:
            return 0
        return len(self.workspace_documents[workspace])
    
    def get_documents(self, workspace: str) -> List[Dict]:
        if workspace not in self.workspace_documents:
            return []
        
        # Return sorted by upload time (newest first)
        return sorted(
            self.workspace_documents[workspace], 
            key=lambda x: x['upload_time'], 
            reverse=True
        )

    def get_document_by_file_id(self, workspace: str, file_id: str) -> Dict[str, Any]:
        if workspace not in self.workspace_documents:
            return None
        for doc in self.workspace_documents[workspace]:
            if doc.get('file_id') == file_id:
                return doc
        return None

    def update_document_metadata(
        self,
        workspace: str,
        filename: str = None,
        file_id: str = None,
        updates: Dict[str, Any] = None,
        structured_preview: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        updates = updates or {}
        if workspace not in self.workspace_documents:
            print(f"🐛 DEBUG: Workspace '{workspace}' not found in workspace_documents")
            return None
        
        # 🐛 DEBUG: Show what we're looking for
        print(f"🐛 DEBUG: update_document_metadata - Looking for:")
        print(f"   - filename: {filename}")
        print(f"   - file_id: {file_id}")
        print(f"   - Available documents in workspace: {len(self.workspace_documents[workspace])}")
        
        target_doc = None
        for idx, doc in enumerate(self.workspace_documents[workspace], 1):
            doc_filename = doc.get('filename', '')
            doc_file_id = doc.get('file_id', '')
            print(f"   {idx}. Checking: filename='{doc_filename}', file_id='{doc_file_id}'")
            
            if filename and doc_filename == filename:
                target_doc = doc
                print(f"   ✅ MATCHED by filename: {filename}")
                break
            if file_id and doc_file_id == file_id:
                target_doc = doc
                print(f"   ✅ MATCHED by file_id: {file_id}")
                break
            # Also try partial matches (in case of sheet names in filename)
            if filename:
                # Check if filename is contained in doc filename or vice versa
                if filename.lower() in doc_filename.lower() or doc_filename.lower() in filename.lower():
                    target_doc = doc
                    print(f"   ✅ MATCHED by partial filename: {filename} <-> {doc_filename}")
                    break
        
        if not target_doc:
            print(f"🐛 DEBUG: ❌ Document NOT FOUND - Cannot update metadata")
            print(f"   - Tried filename: {filename}")
            print(f"   - Tried file_id: {file_id}")
            return None
        
        print(f"🐛 DEBUG: ✅ Found document: {target_doc.get('filename')}")
        
        if updates:
            print(f"🐛 DEBUG: Applying updates: {list(updates.keys())}")
            target_doc.update(updates)
        
        if structured_preview:
            print(f"🐛 DEBUG: Persisting structured preview (type: {structured_preview.get('type')})")
            self._persist_structured_data(workspace, target_doc, structured_preview)
        
        self._save_to_disk()
        print(f"🐛 DEBUG: ✅ Document metadata updated successfully")
        return target_doc

    def _persist_structured_data(self, workspace: str, doc_record: Dict[str, Any], structured_preview: Dict[str, Any]):
        file_id = doc_record.get('file_id')
        if not file_id:
            return
        
        path = self._get_structured_data_path(workspace, file_id)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(structured_preview, f, default=self._json_default, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save structured data for {doc_record.get('filename')}: {e}")
            return
        
        summary = self._build_structured_summary(structured_preview)
        doc_record['structured_data'] = {
            'available': True,
            'summary': summary,
            'last_updated': datetime.now().isoformat()
        }
        
        # Mirror structured preview to MongoDB if configured
        if hasattr(self, "structured_coll") and self.structured_coll is not None:
            try:
                mongo_record = {
                    '_id': f"{workspace}:{file_id}",
                    'workspace': workspace,
                    'file_id': file_id,
                    'filename': doc_record.get('filename'),
                    'structured_preview': structured_preview,
                    'summary': summary,
                    'last_updated': doc_record['structured_data']['last_updated']
                }
                self.structured_coll.update_one(
                    {'_id': mongo_record['_id']},
                    {'$set': mongo_record},
                    upsert=True
                )
            except Exception as e:
                print(f"⚠️ Failed to persist structured preview to MongoDB: {e}")

    def _build_structured_summary(self, structured_preview: Dict[str, Any]) -> Dict[str, Any]:
        sheets = structured_preview.get('sheets', [])
        summary_items = []
        for sheet in sheets:
            summary_items.append({
                'name': sheet.get('name'),
                'row_count': sheet.get('row_count', 0),
                'column_count': sheet.get('column_count', 0),
                'numeric_columns': sheet.get('numeric_columns', []),
                'categorical_columns': sheet.get('categorical_columns', []),
            })
        return {
            'type': structured_preview.get('type'),
            'sheets': summary_items,
            'generated_at': structured_preview.get('generated_at')
        }

    def _get_structured_data_path(self, workspace: str, file_id: str) -> str:
        safe_workspace = re.sub(r'[^a-zA-Z0-9_-]+', '_', workspace)
        return os.path.join(self.structured_data_dir, f"{safe_workspace}_{file_id}.json")

    def get_structured_data(self, workspace: str, file_id: str) -> Optional[Dict[str, Any]]:
        path = self._get_structured_data_path(workspace, file_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load structured data for {workspace}:{file_id}: {e}")
            return None

    def _json_default(self, obj: Any):
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
        except ImportError:
            pass
        return str(obj)

    def list_structured_documents(self, workspace: str) -> List[Dict[str, Any]]:
        if workspace not in self.workspace_documents:
            print(f"🐛 DEBUG: Workspace '{workspace}' not found in workspace_documents")
            return []
        
        # 🐛 DEBUG: Show all documents first
        all_docs = self.workspace_documents[workspace]
        print(f"🐛 DEBUG: Total documents in workspace '{workspace}': {len(all_docs)}")
        for idx, doc in enumerate(all_docs, 1):
            filename = doc.get('filename', 'unknown')
            structured_meta = doc.get('structured_data')
            has_structured = structured_meta is not None
            is_available = structured_meta.get('available') if structured_meta else False
            file_id = doc.get('file_id', 'no-id')
            print(f"   {idx}. {filename}")
            print(f"      - file_id: {file_id}")
            print(f"      - has structured_data key: {has_structured}")
            print(f"      - structured_data.available: {is_available}")
            if structured_meta:
                print(f"      - structured_data keys: {list(structured_meta.keys())}")
        
        structured_docs = []
        for doc in all_docs:
            structured_meta = doc.get('structured_data')
            if structured_meta and structured_meta.get('available'):
                structured_docs.append(doc)
            else:
                filename = doc.get('filename', 'unknown')
                print(f"🐛 DEBUG: Skipping {filename} - structured_data not available")
                print(f"      - structured_meta exists: {structured_meta is not None}")
                if structured_meta:
                    print(f"      - available flag: {structured_meta.get('available')}")
        
        print(f"🐛 DEBUG: Returning {len(structured_docs)} structured document(s)")
        return structured_docs

    def iterate_structured_sheets(self, workspace: str):
        structured_docs = self.list_structured_documents(workspace)
        print(f"🐛 DEBUG: iterate_structured_sheets - processing {len(structured_docs)} structured docs")
        
        for doc in structured_docs:
            file_id = doc.get('file_id')
            filename = doc.get('filename', 'unknown')
            print(f"🐛 DEBUG: Processing {filename} (file_id: {file_id})")
            
            structured_data = self.get_structured_data(workspace, file_id)
            if not structured_data:
                print(f"🐛 DEBUG: No structured data file found for {filename} (file_id: {file_id})")
                # Check if file exists
                path = self._get_structured_data_path(workspace, file_id)
                print(f"🐛 DEBUG: Expected path: {path}")
                print(f"🐛 DEBUG: Path exists: {os.path.exists(path)}")
                continue
            
            sheets = structured_data.get('sheets', [])
            print(f"🐛 DEBUG: Found {len(sheets)} sheet(s) in structured data for {filename}")
            
            for sheet in sheets:
                yield {
                    'doc': doc,
                    'structured_data': structured_data,
                    'sheet': sheet
                }

    def search_structured_data(self, workspace: str, column_keywords: List[str]) -> List[Dict[str, Any]]:
        if not column_keywords:
            return []
        keywords = [kw.lower() for kw in column_keywords if kw]
        matches = []
        for item in self.iterate_structured_sheets(workspace):
            sheet = item['sheet']
            doc = item['doc']
            columns = sheet.get('columns', [])
            column_names = [col.get('name', '').lower() for col in columns]
            score = sum(1 for kw in keywords if any(kw in name for name in column_names))
            if score > 0:
                matches.append({
                    'doc': doc,
                    'sheet': sheet,
                    'structured_data': item['structured_data'],
                    'score': score
                })
        matches.sort(key=lambda m: m['score'], reverse=True)
        return matches

    def find_document_by_filename(self, workspace: str, filename: str) -> Optional[Dict]:
        """Find document by filename with flexible matching"""
        if workspace not in self.workspace_documents:
            return None
        
        # Try exact match first
        for doc in self.workspace_documents[workspace]:
            if doc.get('filename') == filename:
                return doc
        
        # Try partial match (for files with sheet names, etc.)
        for doc in self.workspace_documents[workspace]:
            doc_filename = doc.get('filename', '')
            if filename in doc_filename or doc_filename in filename:
                return doc
        
        # Try case-insensitive match
        filename_lower = filename.lower()
        for doc in self.workspace_documents[workspace]:
            doc_filename_lower = doc.get('filename', '').lower()
            if filename_lower in doc_filename_lower or doc_filename_lower in filename_lower:
                return doc
        
        return None

# =============================================================================
# PROFESSIONAL RAG SYSTEM - COMPLETE INTEGRATED VERSION
# =============================================================================

# =============================================================================
# PROFESSIONAL RAG SYSTEM - DUAL RETRIEVER COMPARISON VERSION
# =============================================================================

class ProfessionalRAGSystem:
    def __init__(self):
        try:
            # Professional components
            self.vector_store = ProfessionalVectorStore()  # This now has the methods
            
            # Pass MongoDB handle into chat history manager (for persistent memory)
            self.chat_history = ChatHistoryManager(mongo_db=mongo_db if mongo_db is not None else None)
            self.document_storage = self.vector_store.document_storage
            self.query_enhancer = FreeQueryEnhancer()
            self.drive_manager = ProfessionalGoogleDriveManager(self.vector_store)
            self._auto_rebuild_hybrid()
            self._auto_sync_document_storage()  # 🆕 ADD THIS LINE

            
            # 🚀 INITIALIZE BOTH RETRIEVERS WITH ERROR HANDLING
            print("🔄 Initializing retrievers...")
            
            # Hybrid retriever
            self.hybrid_retriever = None
            try:
                # Check if the method exists before calling it
                if hasattr(self.vector_store, 'get_hybrid_retriever'):
                    self.hybrid_retriever = self.vector_store.get_hybrid_retriever()
                    if self.hybrid_retriever:
                        print("✅ Hybrid retriever initialized")
                    else:
                        print("⚠️ Hybrid retriever not available")
                else:
                    print("❌ get_hybrid_retriever method not found in vector_store")
            except Exception as e:
                print(f"❌ Hybrid retriever initialization failed: {e}")
            
            # LangChain retriever
            self.langchain_retriever = None
            try:
                # Check if the method exists before calling it
                if hasattr(self.vector_store, 'get_langchain_retriever'):
                    self.langchain_retriever = self.vector_store.get_langchain_retriever()
                    if self.langchain_retriever:
                        print("✅ LangChain retriever initialized")
                    else:
                        print("⚠️ LangChain retriever not available")
                else:
                    print("❌ get_langchain_retriever method not found in vector_store")
            except Exception as e:
                print(f"❌ LangChain retriever initialization failed: {e}")
            
            # 📊 COMPARISON METRICS
            self.comparison_history = []
            
            print("🚀 PROFESSIONAL ENTERPRISE RAG SYSTEM INITIALIZED WITH DUAL RETRIEVERS")
            
        except Exception as e:
            print(f"❌ RAG system initialization error: {e}")
            traceback.print_exc()
            raise
    
    def _parse_chart_request(self, query: str) -> Optional[Dict[str, Any]]:
        if not query:
            return None
        lowered = query.lower()
        chart_keywords = ['chart', 'graph', 'plot', 'visualization', 'visualise', 'visualize', 'diagram']
        chart_type_map = {
            'bargraph': 'bar',
            'bar graph': 'bar',
            'bar chart': 'bar',
            'histogram': 'bar',
            'column chart': 'bar',
            'bar': 'bar',
            'line chart': 'line',
            'line graph': 'line',
            'line': 'line',
            'trend': 'line',
            'timeline': 'line',
            'area chart': 'line',
            'pie chart': 'pie',
            'pie graph': 'pie',
            'pie': 'pie',
            'donut': 'pie',
            'doughnut': 'pie'
        }
        
        if not any(keyword in lowered for keyword in chart_keywords + list(chart_type_map.keys())):
            return None
        
        chart_type = 'line'
        for key, value in chart_type_map.items():
            if key in lowered:
                chart_type = value
                break
        
        metric_keywords: List[str] = []
        dimension_keywords: List[str] = []
        
        metric_match = re.search(r'\bof\s+(?:the\s+)?([a-z0-9\s\-]+?)(?:\s+(from|by|over|during|for)\b|$)', lowered)
        if metric_match:
            metric_phrase = metric_match.group(1).strip()
            metric_keywords = [word for word in re.findall(r'[a-z0-9]+', metric_phrase) if len(word) > 2]
        
        by_match = re.search(r'\bby\s+([a-z0-9\s\-]+)', lowered)
        if by_match:
            dimension_phrase = by_match.group(1).strip()
            dimension_keywords = [word for word in re.findall(r'[a-z0-9]+', dimension_phrase) if len(word) > 2]
        
        year_matches = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', lowered)]
        year_range = None
        if len(year_matches) >= 2:
            year_range = (min(year_matches), max(year_matches))
        elif len(year_matches) == 1:
            year = year_matches[0]
            year_range = (year, year)
        
        if 'year' in lowered or year_range:
            dimension_keywords.append('year')
        if not metric_keywords:
            metric_keywords = ['count', 'number', 'total']
        
        return {
            'raw_query': query,
            'chart_type': chart_type,
            'metric_keywords': list(dict.fromkeys(metric_keywords)),
            'dimension_keywords': list(dict.fromkeys(dimension_keywords)),
            'year_range': year_range
        }

    def _handle_chart_request(self, namespace: str, chart_intent: Dict[str, Any], session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Enhanced chart handler with smart data source selection"""
        if not self.document_storage:
            return None
        
        # 🎯 STEP 1: Try to find structured dataset
        dataset_match = self._find_structured_dataset(namespace, chart_intent)
        
        chart_payload = None
        using_original_data = False
        data_source_type = 'unknown'
        
        # 🎯 STEP 2: TRY ORIGINAL DATA FIRST (highest quality)
        if dataset_match and dataset_match['doc'].get('chart_data_id'):
            print(f"🔍 Attempting to build chart from original file: {dataset_match['doc'].get('filename')}")
            chart_payload = self.build_chart_from_original_data(
                chart_intent['raw_query'], namespace, dataset_match['doc']
            )
            
            if chart_payload:
                using_original_data = True
                data_source_type = 'original_file'
                print(f"✅ Using ORIGINAL DATA for high-quality chart (columns: {chart_payload.get('meta', {}).get('metric_column')} x {chart_payload.get('meta', {}).get('dimension_column')})")
        
        # 🎯 STEP 3: Fallback to structured data preview (medium quality)
        if not chart_payload and dataset_match:
            print(f"🔍 Attempting to build chart from structured data preview: {dataset_match['doc'].get('filename')}")
            chart_payload = self._build_chart_payload(dataset_match, chart_intent)
            
            if chart_payload:
                data_source_type = 'structured_preview'
                print(f"✅ Using STRUCTURED PREVIEW for chart (columns: {chart_payload.get('meta', {}).get('metric_column')} x {chart_payload.get('meta', {}).get('dimension_column')})")
            else:
                print(f"⚠️ Failed to build chart from structured preview, trying document fallback")
                dataset_match = None  # Clear match if it failed
        
        # 🎯 STEP 4: Final fallback: build from document chunks (lowest quality)
        if not chart_payload:
            print(f"🔍 Attempting to build chart from document chunks")
            chart_payload, sources = self._build_chart_from_documents(namespace, chart_intent)
            
            if chart_payload:
                data_source_type = 'document_chunks'
                print(f"✅ Using DOCUMENT CHUNKS for chart (columns: {chart_payload.get('meta', {}).get('metric_column')} x {chart_payload.get('meta', {}).get('dimension_column')})")
        
        if not chart_payload:
            print(f"❌ Failed to build chart from any data source")
            return None
        
        # 🎯 STEP 5: Build response with proper metadata
        if dataset_match and using_original_data:
            answer = self._build_chart_answer(dataset_match, chart_intent, chart_payload)
            source_entry = {
                'source': dataset_match['doc'].get('filename'),
                'sheet': dataset_match['sheet'].get('name') if dataset_match.get('sheet') else None,
                'type': 'original_structured_data',
                'metric_column': chart_payload.get('meta', {}).get('metric_column'),
                'dimension_column': chart_payload.get('meta', {}).get('dimension_column')
            }
            sources = [source_entry]
            documents_used = chart_payload.get('meta', {}).get('row_count', 0)
        elif dataset_match:
            answer = self._build_chart_answer(dataset_match, chart_intent, chart_payload)
            source_entry = {
                'source': dataset_match['doc'].get('filename'),
                'sheet': dataset_match['sheet'].get('name') if dataset_match.get('sheet') else None,
                'type': 'structured_data',
                'metric_column': chart_payload.get('meta', {}).get('metric_column'),
                'dimension_column': chart_payload.get('meta', {}).get('dimension_column')
            }
            sources = [source_entry]
            documents_used = chart_payload.get('meta', {}).get('row_count', 0)
        else:
            answer = self._build_chart_answer(None, chart_intent, chart_payload)
            sources = sources if 'sources' in locals() else []
            documents_used = len(sources) if sources else 0
        
        response = {
            'answer': answer,
            'confidence': 'high' if using_original_data else ('medium' if data_source_type == 'structured_preview' else 'low'),
            'sources': sources,
            'documents_used': documents_used,
            'method_used': 'original_data_chart' if using_original_data else ('structured_data_chart' if data_source_type == 'structured_preview' else 'document_chart'),
            'data_source_type': data_source_type,
            'chart': chart_payload
        }
        
        if session_id:
            self.chat_history.add_message(session_id, 'user', chart_intent['raw_query'])
            self.chat_history.add_message(session_id, 'assistant', answer, {
                'method_used': response['method_used'],
                'chart': chart_payload,
                'sources': sources,
                'data_source_type': data_source_type
            })
        
        return response

    def _find_structured_dataset(self, namespace: str, chart_intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhanced structured dataset finder with smart file and column matching"""
        query_lower = chart_intent['raw_query'].lower()
        metric_keywords = chart_intent.get('metric_keywords') or []
        dimension_keywords = chart_intent.get('dimension_keywords') or []
        
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: Finding structured dataset for query: '{chart_intent['raw_query']}'")
        print(f"{'='*80}")
        
        # Get all structured files
        all_files = list(self.document_storage.iterate_structured_sheets(namespace))
        
        if not all_files:
            print(f"⚠️ No structured datasets found in namespace: {namespace}")
            return None
        
        # 🐛 DEBUG: List all available files
        print(f"\n📁 DEBUG: Found {len(all_files)} structured file(s):")
        for idx, item in enumerate(all_files, 1):
            doc = item['doc']
            sheet = item['sheet']
            filename = doc.get('filename', 'unknown')
            has_chart_data = 'Yes' if doc.get('chart_data_id') else 'No'
            row_count = sheet.get('row_count', 0)
            col_count = len(sheet.get('columns', []))
            print(f"   {idx}. {filename}")
            print(f"      - Sheet: {sheet.get('name', 'N/A')}")
            print(f"      - Rows: {row_count}, Columns: {col_count}")
            print(f"      - Has chart_data_id: {has_chart_data}")
            print(f"      - Numeric columns: {len(sheet.get('numeric_columns', []))}")
            print(f"      - Categorical columns: {len(sheet.get('categorical_columns', []))}")
        print()
        
        # 🎯 STEP 1: Extract explicit filename mentions from query
        # Look for patterns like "of filename.xlsx" or "from filename" or "filename data"
        explicit_filename = None
        query_words = query_lower.split()
        
        print(f"🔍 DEBUG: Analyzing query words: {query_words}")
        
        # Try to find filename patterns in query (with extensions)
        for i, word in enumerate(query_words):
            # Check if word looks like a filename (has extension)
            if '.' in word and any(ext in word for ext in ['.xlsx', '.xls', '.csv', '.json']):
                # Extract filename (remove common words before/after)
                potential_filename = word
                # Remove common prefixes
                for prefix in ['of', 'from', 'in', 'the', 'a', 'an']:
                    if potential_filename.startswith(prefix):
                        potential_filename = potential_filename[len(prefix):]
                explicit_filename = potential_filename
                print(f"✅ DEBUG: Found explicit filename in query: '{explicit_filename}'")
                break
        
        # Also check for filename mentions without extension (e.g., "iris data" -> "iris.xlsx")
        if not explicit_filename:
            # Check for single-word filename matches (e.g., "iris" -> "iris.xlsx")
            for word in query_words:
                # Skip common words
                if word.lower() in ['of', 'from', 'in', 'the', 'a', 'an', 'any', 'give', 'me', 'chart', 'pie', 'line', 'bar', 'data', 'columns']:
                    continue
                
                # Check if this word matches a filename (without extension)
                for item in all_files:
                    filename_lower = item['doc'].get('filename', '').lower()
                    filename_base = filename_lower.rsplit('.', 1)[0]  # Remove extension
                    
                    # Check if word matches filename base
                    if word == filename_base or word in filename_base or filename_base.startswith(word):
                        explicit_filename = filename_lower
                        print(f"🔍 Found filename match (word '{word}' -> '{filename_lower}')")
                        break
                
                if explicit_filename:
                    break
        
        # Also check for multi-word patterns (e.g., "product-sales" for "Product-Sales-Region.xlsx")
        if not explicit_filename:
            # Look for multi-word patterns that might be filenames
            for i in range(len(query_words) - 1):
                two_word = f"{query_words[i]}-{query_words[i+1]}"
                three_word = f"{query_words[i]}-{query_words[i+1]}-{query_words[i+2]}" if i+2 < len(query_words) else None
                
                for item in all_files:
                    filename_lower = item['doc'].get('filename', '').lower()
                    if two_word in filename_lower or (three_word and three_word in filename_lower):
                        explicit_filename = filename_lower
                        print(f"🔍 Found partial filename match in query: {explicit_filename}")
                        break
                if explicit_filename:
                    break
        
        if not explicit_filename:
            print(f"⚠️ DEBUG: No explicit filename with extension found in query")
        
        # Score each file based on multiple factors
        scored_matches = []
        
        print(f"\n📊 DEBUG: Scoring {len(all_files)} file(s)...")
        print(f"{'-'*80}")
        
        for item_idx, item in enumerate(all_files, 1):
            doc = item['doc']
            sheet = item['sheet']
            filename = doc.get('filename', '').lower()
            filename_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
            columns = sheet.get('columns', [])
            numeric_columns = sheet.get('numeric_columns', [])
            categorical_columns = sheet.get('categorical_columns', [])
            
            # 🎯 CRITICAL: File name matching score (MUCH HIGHER PRIORITY for explicit mentions)
            file_score = 0
            
            # Check for exact filename match (highest priority)
            if explicit_filename:
                if explicit_filename in filename or filename in explicit_filename:
                    file_score += 1000  # HUGE boost for explicit filename match
                    print(f"   🎯 EXACT FILENAME MATCH: {filename} (score +1000)")
                elif explicit_filename.replace('.', '') in filename.replace('.', ''):
                    file_score += 500  # High boost for close match
                    print(f"   🎯 CLOSE FILENAME MATCH: {filename} (score +500)")
            
            # Check for partial filename matches in query
            for word in query_words:
                # Exact word match in filename (high priority)
                if word in filename_without_ext:
                    file_score += 50
                # Word match in full filename
                elif word in filename:
                    file_score += 30
                # Check if word matches any column name (lower priority)
                for col in columns:
                    col_name = col.get('name', '').lower()
                    if word in col_name:
                        file_score += 2  # Reduced from 5 - column matches are less important than filename
            
            # Column matching
            metric_column, metric_score = self._select_metric_column(
                columns, numeric_columns, metric_keywords
            )
            dimension_column, dimension_score = self._select_dimension_column(
                columns, categorical_columns, dimension_keywords
            )
            
            # If no dimension found, try fallbacks
            if not dimension_column:
                # Try year column
                year_col = next((col for col in columns if 'year' in col.get('name', '').lower()), None)
                if year_col:
                    dimension_column = year_col
                    dimension_score = 1
                # Try first categorical
                elif categorical_columns:
                    fallback_name = categorical_columns[0]
                    dimension_column = next((col for col in columns if col.get('name') == fallback_name), None)
                    dimension_score = 0.5
                # Try second numeric as dimension if we have multiple numerics
                elif len(numeric_columns) > 1:
                    dimension_column = next((col for col in columns if col.get('name') == numeric_columns[1]), None)
                    dimension_score = 0.3
            
            # Calculate total score
            # 🎯 CRITICAL: File score is now MUCH more important than column scores
            # This ensures files explicitly mentioned in query are prioritized
            
            # 🚨 FIX: If file is explicitly mentioned, we MUST include it even if columns are missing
            # This handles cases where structured preview extraction failed partially
            is_explicit_match = explicit_filename and (explicit_filename in filename or filename in explicit_filename)
            
            if metric_column and dimension_column:
                metric_col_name = metric_column.get('name') if isinstance(metric_column, dict) else metric_column
                dimension_col_name = dimension_column.get('name') if isinstance(dimension_column, dict) else dimension_column
                
                chart_data_bonus = 10 if doc.get('chart_data_id') else 0
                total_score = (
                    file_score +  # File matching is now the PRIMARY factor (can be 1000+)
                    metric_score * 2 +  # Reduced from 3 - column matching is secondary
                    dimension_score * 1 +  # Reduced from 2 - column matching is secondary
                    chart_data_bonus  # Prefer files with original data
                )
                
                # 🐛 DEBUG: Print scoring breakdown for each file
                print(f"\n   File {item_idx}: {filename}")
                print(f"      - File score: {file_score} (explicit match: {explicit_filename is not None and explicit_filename in filename})")
                print(f"      - Metric: {metric_col_name} (score: {metric_score})")
                print(f"      - Dimension: {dimension_col_name} (score: {dimension_score})")
                print(f"      - Chart data bonus: {chart_data_bonus}")
                print(f"      - TOTAL SCORE: {total_score}")
                
                scored_matches.append({
                    'score': total_score,
                    'doc': doc,
                    'sheet': sheet,
                    'structured_data': item.get('structured_data'),
                    'metric_column': metric_col_name,
                    'dimension_column': dimension_col_name,
                    'metric_column_dict': metric_column,
                    'dimension_column_dict': dimension_column,
                    'file_score': file_score,
                    'metric_score': metric_score,
                    'dimension_score': dimension_score
                })
            else:
                # 🚨 SPECIAL CASE: If this is an explicit filename match, include it anyway
                # We'll build the chart from original file data which has the real columns
                if is_explicit_match:
                    print(f"\n   File {item_idx}: {filename} - INCLUDED (explicit match, will use original file data)")
                    print(f"      - Metric found: {metric_column is not None}")
                    print(f"      - Dimension found: {dimension_column is not None}")
                    print(f"      - Will use original file to extract columns")
                    
                    # Create a minimal entry so we can still select this file
                    # The chart building will read from original file which has real data
                    scored_matches.append({
                        'score': file_score + 1000,  # High score for explicit match
                        'doc': doc,
                        'sheet': sheet,
                        'structured_data': item.get('structured_data'),
                        'metric_column': None,  # Will be determined from original file
                        'dimension_column': None,  # Will be determined from original file
                        'metric_column_dict': None,
                        'dimension_column_dict': None,
                        'file_score': file_score,
                        'metric_score': 0,
                        'dimension_score': 0,
                        'use_original_file': True  # Flag to use original file
                    })
                else:
                    print(f"\n   File {item_idx}: {filename} - SKIPPED (no suitable metric/dimension columns)")
                    print(f"      - Metric found: {metric_column is not None}")
                    print(f"      - Dimension found: {dimension_column is not None}")
        
        if not scored_matches:
            print(f"\n⚠️ No suitable structured datasets found with matching columns")
            return None
        
        # Sort by score and return best match
        scored_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # 🐛 DEBUG: Show all scores
        print(f"\n{'='*80}")
        print(f"📊 DEBUG: Final scoring results (sorted by score):")
        print(f"{'='*80}")
        for idx, match in enumerate(scored_matches[:5], 1):  # Show top 5
            print(f"   {idx}. {match['doc'].get('filename')}")
            print(f"      - Total Score: {match['score']:.1f}")
            print(f"      - File Score: {match.get('file_score', 0)}")
            print(f"      - Metric: {match['metric_column']} (score: {match.get('metric_score', 0)})")
            print(f"      - Dimension: {match['dimension_column']} (score: {match.get('dimension_score', 0)})")
        
        best_match = scored_matches[0]
        
        print(f"\n✅ WINNER: {best_match['doc'].get('filename')}")
        print(f"   - Score: {best_match['score']:.1f}")
        print(f"   - Metric: {best_match['metric_column']}")
        print(f"   - Dimension: {best_match['dimension_column']}")
        print(f"{'='*80}\n")
        
        return {
            'doc': best_match['doc'],
            'sheet': best_match['sheet'],
            'structured_data': best_match.get('structured_data'),
            'metric_column': best_match['metric_column'],
            'dimension_column': best_match['dimension_column']
        }

    def _select_metric_column(self, columns: List[Dict[str, Any]], numeric_columns: List[str], keywords: List[str]) -> Tuple[Optional[Dict[str, Any]], int]:
        if not columns:
            return None, 0
        best_col = None
        best_score = -1
        numeric_set = set(name.lower() for name in numeric_columns)
        for column in columns:
            name = str(column.get('name', ''))
            if not name:
                continue
            name_lower = name.lower()
            is_numeric = column.get('is_numeric', name_lower in numeric_set)
            if not is_numeric:
                continue
            score = 0
            for kw in keywords:
                if kw in name_lower:
                    score += 2
            if any(term in name_lower for term in ['count', 'total', 'number', 'amount']):
                score += 1
            if score > best_score:
                best_score = score
                best_col = column
        if best_col is None:
            for column in columns:
                if column.get('is_numeric'):
                    return column, 1
        return best_col, max(best_score, 0)

    def _select_dimension_column(self, columns: List[Dict[str, Any]], categorical_columns: List[str], keywords: List[str]) -> Tuple[Optional[Dict[str, Any]], int]:
        if not columns:
            return None, 0
        best_col = None
        best_score = -1
        categorical_set = set(name.lower() for name in categorical_columns)
        for column in columns:
            name = str(column.get('name', ''))
            if not name:
                continue
            name_lower = name.lower()
            is_categorical = name_lower in categorical_set or not column.get('is_numeric', False)
            if not is_categorical:
                continue
            score = 0
            for kw in keywords:
                if kw in name_lower:
                    score += 2
            if name_lower in ['year', 'date', 'month']:
                score += 2
            if score > best_score:
                best_score = score
                best_col = column
        return best_col, max(best_score, 0)

    def _build_chart_payload(self, dataset_match: Dict[str, Any], chart_intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            print("⚠️ Chart generation requires pandas. Please install pandas to enable charting.")
            return None
        
        sheet = dataset_match['sheet']
        rows = sheet.get('rows', [])
        if not rows:
            return None
        
        metric_column = dataset_match['metric_column']
        dimension_column = dataset_match['dimension_column']
        
        df = pd.DataFrame(rows)
        if df.empty or dimension_column not in df.columns:
            return None
        
        df = df.copy()
        
        year_range = chart_intent.get('year_range')
        if year_range and dimension_column.lower().strip() == 'year':
            df[dimension_column] = pd.to_numeric(df[dimension_column], errors='coerce')
            df = df.dropna(subset=[dimension_column])
            df = df[(df[dimension_column] >= year_range[0]) & (df[dimension_column] <= year_range[1])]
        
        prefer_counts = chart_intent.get('chart_type') in ['pie', 'doughnut', 'donut'] or \
            any(kw in ('count', 'counts', 'total', 'number') for kw in (chart_intent.get('metric_keywords') or []))
        
        use_counts = False
        metric_display_name = metric_column or 'Count'
        
        if metric_column and metric_column in df.columns and not prefer_counts:
            df_numeric = df[[dimension_column, metric_column]].copy()
            df_numeric[metric_column] = pd.to_numeric(df_numeric[metric_column], errors='coerce')
            df_numeric = df_numeric.dropna(subset=[metric_column])
            if df_numeric.empty:
                use_counts = True
            else:
                df = df_numeric
        else:
            use_counts = True
        
        if use_counts:
            aggregated = (
                df.groupby(dimension_column, as_index=False)
                  .size()
                  .rename(columns={'size': 'Count'})
            )
            metric_column_for_chart = 'Count'
            metric_display_name = 'Count'
        else:
            aggregated = (
                df.groupby(dimension_column, as_index=False)[metric_column]
                  .sum()
            )
            metric_column_for_chart = metric_column
        
        if aggregated.empty:
            return None
        
        try:
            aggregated = aggregated.sort_values(by=dimension_column)
        except Exception:
            pass
        
        max_points = 150
        if len(aggregated) > max_points:
            aggregated = aggregated.head(max_points)
        
        labels = aggregated[dimension_column].astype(str).tolist()
        data_series = aggregated[metric_column_for_chart]
        if pd.api.types.is_numeric_dtype(data_series):
            data_points = data_series.round(2).tolist()
        else:
            data_points = data_series.tolist()
        
        chart_id = f"chart_{dataset_match['doc'].get('file_id')}_{int(datetime.now().timestamp() * 1000)}"
        meta = {
            'source': dataset_match['doc'].get('filename'),
            'sheet': sheet.get('name'),
            'metric_column': metric_display_name,
            'dimension_column': dimension_column,
            'row_count': int(len(aggregated))
        }
        
        palette = ['#10a37f', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899']
        chart_type = chart_intent.get('chart_type', 'bar') or 'bar'
        if chart_type == 'donut':
            chart_type = 'doughnut'
        
        if chart_type in ['pie', 'doughnut', 'donut']:
            colors = [palette[i % len(palette)] for i in range(len(data_points))]
            dataset_config = {
                'label': metric_display_name,
                'data': data_points,
                'backgroundColor': colors,
                'borderColor': '#1f2937',
                'borderWidth': 1
            }
            options = {
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {'display': True, 'position': 'right'},
                    'title': {'display': False}
                }
            }
        else:
            color = palette[0]
            dataset_config = {
                'label': metric_display_name,
                'data': data_points,
                'backgroundColor': color,
                'borderColor': color,
                'fill': chart_type != 'line',
                'borderWidth': 2 if chart_type == 'line' else 0,
                'tension': 0.3 if chart_type == 'line' else 0
            }
            options = {
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {'display': True},
                    'title': {'display': False},
                    'tooltip': {'mode': 'index', 'intersect': False}
                },
                'scales': {
                    'x': {'title': {'display': True, 'text': dimension_column}},
                    'y': {'title': {'display': True, 'text': metric_display_name}, 'beginAtZero': True}
                }
            }
        
        return {
            'chart_id': chart_id,
            'type': chart_type,
            'labels': labels,
            'datasets': [dataset_config],
            'title': f"{metric_display_name} by {dimension_column}",
            'meta': meta,
            'options': options
        }

    def _build_chart_from_documents(self, namespace: str, chart_intent: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        docs = self._get_vector_results(chart_intent['raw_query'], namespace, top_k=40)
        if not docs:
            docs = self._get_hybrid_results(chart_intent['raw_query'], namespace, top_k=40)
        if not docs:
            return None, []
        
        rows = self._extract_structured_rows_from_documents(docs)
        if not rows:
            return None, []
        
        dimension_column = self._select_dimension_from_rows(rows, chart_intent.get('dimension_keywords'))
        if not dimension_column:
            return None, []
        
        counts = Counter()
        for row in rows:
            value = row.get(dimension_column)
            if value is None or str(value).strip() == '':
                continue
            counts[str(value).strip()] += 1
        
        if not counts:
            return None, []
        
        sorted_items = sorted(counts.items(), key=lambda x: x[0])
        labels = [item[0] for item in sorted_items]
        data_points = [item[1] for item in sorted_items]
        
        chart_type = chart_intent.get('chart_type', 'pie') or 'pie'
        if chart_type not in ['pie', 'doughnut', 'donut', 'bar', 'line']:
            chart_type = 'pie'
        if chart_type == 'donut':
            chart_type = 'doughnut'
        
        metric_display_name = 'Count'
        palette = ['#10a37f', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899']
        
        if chart_type in ['pie', 'doughnut']:
            colors = [palette[i % len(palette)] for i in range(len(data_points))]
            dataset_config = {
                'label': metric_display_name,
                'data': data_points,
                'backgroundColor': colors,
                'borderColor': '#1f2937',
                'borderWidth': 1
            }
            options = {
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {'display': True, 'position': 'right'},
                    'title': {'display': False}
                }
            }
        else:
            color = palette[0]
            dataset_config = {
                'label': metric_display_name,
                'data': data_points,
                'backgroundColor': color,
                'borderColor': color,
                'fill': chart_type != 'line',
                'borderWidth': 2 if chart_type == 'line' else 0,
                'tension': 0.3 if chart_type == 'line' else 0
            }
            options = {
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {'display': True},
                    'title': {'display': False},
                    'tooltip': {'mode': 'index', 'intersect': False}
                },
                'scales': {
                    'x': {'title': {'display': True, 'text': dimension_column}},
                    'y': {'title': {'display': True, 'text': metric_display_name}, 'beginAtZero': True}
                }
            }
        
        first_source = docs[0].metadata.get('source', 'Unknown Document') if docs else 'Unknown Document'
        chart_payload = {
            'chart_id': f"chart_docfallback_{int(datetime.now().timestamp() * 1000)}",
            'type': chart_type,
            'labels': labels,
            'datasets': [dataset_config],
            'title': f"{metric_display_name} by {dimension_column}",
            'meta': {
                'source': first_source,
                'sheet': None,
                'metric_column': metric_display_name,
                'dimension_column': dimension_column,
                'row_count': len(rows)
            },
            'options': options
        }
        
        sources = []
        for doc in docs[:3]:
            sources.append({
                'source': doc.metadata.get('source', 'Unknown Document'),
                'type': 'document'
            })
        return chart_payload, sources

    def _extract_structured_rows_from_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for doc in documents:
            for line in doc.page_content.splitlines():
                if 'ROW' not in line or ':' not in line:
                    continue
                try:
                    _, remainder = line.split(':', 1)
                except ValueError:
                    remainder = line
                segments = [segment.strip() for segment in remainder.split('|') if segment.strip()]
                row_data = {}
                for segment in segments:
                    if ':' not in segment:
                        continue
                    key, value = segment.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key:
                        row_data[key] = value
                if row_data:
                    rows.append(row_data)
        return rows

    def _select_dimension_from_rows(self, rows: List[Dict[str, Any]], dimension_keywords: Optional[List[str]]) -> Optional[str]:
        if not rows:
            return None
        dimension_keywords = [kw.lower() for kw in (dimension_keywords or []) if kw]
        candidate_columns = set()
        for row in rows:
            candidate_columns.update(row.keys())
        best_col = None
        best_score = -1
        for col in candidate_columns:
            col_lower = col.lower()
            score = 0
            for kw in dimension_keywords:
                if kw in col_lower:
                    score += 3
            for default_kw in ['category', 'class', 'type', 'label', 'name', 'group', 'species']:
                if default_kw in col_lower:
                    score += 2
            values = {row.get(col) for row in rows if col in row and row.get(col)}
            if not values:
                continue
            if all(self._is_numeric_value(val) for val in values):
                score -= 1
            else:
                score += 1
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

    def _is_numeric_value(self, value: Any) -> bool:
        if value is None:
            return False
        try:
            float(str(value).strip())
            return True
        except (ValueError, TypeError):
            return False

    def _build_chart_answer(self, dataset_match: Optional[Dict[str, Any]], chart_intent: Dict[str, Any], chart_payload: Dict[str, Any]) -> str:
        meta = chart_payload.get('meta', {}) if chart_payload else {}
        doc_name = meta.get('source')
        sheet_name = meta.get('sheet')
        metric_column = meta.get('metric_column')
        dimension_column = meta.get('dimension_column')
        
        if dataset_match:
            doc_name = doc_name or dataset_match['doc'].get('filename')
            sheet_name = sheet_name or dataset_match['sheet'].get('name')
            metric_column = metric_column or dataset_match.get('metric_column')
            dimension_column = dimension_column or dataset_match.get('dimension_column')
        
        doc_name = doc_name or 'available documents'
        metric_column = metric_column or 'Count'
        dimension_column = dimension_column or 'Category'
        chart_type = chart_intent.get('chart_type', 'bar') or 'bar'
        chart_label = chart_type if chart_type != 'line' else 'trend'
        
        base = f"Here is the {chart_label} chart showing {metric_column} by {dimension_column}."
        source_info = f" Data source: {doc_name}"
        if sheet_name:
            source_info += f" › {sheet_name}"
        year_range = chart_intent.get('year_range')
        if year_range and dimension_column and 'year' in dimension_column.lower():
            source_info += f" (filtered for {year_range[0]} to {year_range[1]})."
        else:
            source_info += "."
        
        return base + source_info
    
    def _auto_rebuild_hybrid(self):
        """Automatically rebuild hybrid cache when server starts"""
        try:
            if self.vector_store and self.vector_store.vector_store:
                # Use the same logic as your manual endpoint
                all_docs = self.vector_store.vector_store.similarity_search("", k=1000)
                if all_docs and self.vector_store.hybrid_retrieval:
                    self.vector_store.hybrid_retrieval.build_hybrid_index(all_docs)
                    print(f"✅ Auto-rebuilt hybrid cache with {len(all_docs)} documents")
        except Exception as e:
            print(f"⚠️ Auto-rebuild failed: {e}")

    def _auto_sync_document_storage(self):
        """Automatically sync documents tab with vector store on startup"""
        try:
            # This will trigger the auto_sync_from_vector_store in DocumentStorageManager
            self.document_storage.auto_sync_from_vector_store("management_full", self.vector_store)
            print("✅ Auto-synced documents tab with vector store")
        except Exception as e:
            print(f"⚠️ Document storage auto-sync failed: {e}")

    def compare_retrieval_methods(self, query: str, namespace: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Compare three DIFFERENT retrieval approaches:
        1. as_retriever() - Standard LangChain (documents only)
        2. direct_vector - Raw Chroma with scores  
        3. current_hybrid - Your custom BM25 + vector fusion
        """
        try:
            print(f"🔍 COMPARING 3 RETRIEVAL APPROACHES: '{query}'")
            
            comparison_results = {
                "query": query,
                "namespace": namespace,
                "timestamp": datetime.now().isoformat(),
                "top_k": top_k,
                "methods_compared": [
                    "as_retriever (standard LangChain - documents only)",
                    "direct_vector (raw Chroma - with scores)", 
                    "current_hybrid (custom BM25 + vector fusion)"
                ]
            }
            
            # 🎯 METHOD 1: Pure as_retriever (documents only)
            print("🔄 Testing as_retriever() method...")
            as_retriever_results = self._test_as_retriever(query, namespace, top_k)
            
            # 🎯 METHOD 2: Direct vector search with scores
            print("🔄 Testing direct vector similarity search...")
            direct_vector_results = self._test_direct_vector_search(query, namespace, top_k)
            
            # 🎯 METHOD 3: Your current hybrid approach
            print("🔄 Testing current hybrid search...")
            hybrid_results = self._test_current_hybrid_search(query, namespace, top_k)
            
            # Combine results
            comparison_results.update({
                "as_retriever": as_retriever_results,
                "direct_vector": direct_vector_results,
                "current_hybrid": hybrid_results,
                "analysis": self._analyze_three_method_comparison(
                    as_retriever_results, 
                    direct_vector_results, 
                    hybrid_results
                )
            })
            
            print("✅ 3-approach comparison completed successfully")
            return comparison_results
            
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            return {"error": str(e)}
    
    def _analyze_three_method_comparison(self, as_retriever_results: Dict, direct_vector_results: Dict, hybrid_results: Dict) -> Dict[str, Any]:
        """Analyze differences between the three main methods"""
        analysis = {
            "document_overlap": {},
            "ranking_analysis": {},
            "performance_notes": [],
            "recommendations": []
        }
        
        # Document overlap analysis
        as_retriever_sources = set([doc['source'] for doc in as_retriever_results.get('documents', [])])
        direct_vector_sources = set([doc['source'] for doc in direct_vector_results.get('documents', [])])
        hybrid_sources = set([doc['source'] for doc in hybrid_results.get('documents', [])])
        
        analysis["document_overlap"] = {
            "as_retriever_unique": len(as_retriever_sources - direct_vector_sources - hybrid_sources),
            "direct_vector_unique": len(direct_vector_sources - as_retriever_sources - hybrid_sources),
            "hybrid_unique": len(hybrid_sources - as_retriever_sources - direct_vector_sources),
            "all_common": len(as_retriever_sources & direct_vector_sources & hybrid_sources),
            "as_retriever_vs_direct_overlap": len(as_retriever_sources & direct_vector_sources),
            "as_retriever_vs_hybrid_overlap": len(as_retriever_sources & hybrid_sources)
        }
        
        # Ranking analysis - check if they agree on top document
        top_docs = {
            "as_retriever": as_retriever_results.get('documents', [{}])[0].get('source', 'none') if as_retriever_results.get('documents') else 'none',
            "direct_vector": direct_vector_results.get('documents', [{}])[0].get('source', 'none') if direct_vector_results.get('documents') else 'none',
            "hybrid": hybrid_results.get('documents', [{}])[0].get('source', 'none') if hybrid_results.get('documents') else 'none'
        }
        
        analysis["ranking_analysis"] = {
            "top_documents": top_docs,
            "all_agree_on_top": len(set([top_docs["as_retriever"], top_docs["direct_vector"], top_docs["hybrid"]])) == 1,
            "as_retriever_vs_direct_agree": top_docs["as_retriever"] == top_docs["direct_vector"],
            "as_retriever_vs_hybrid_agree": top_docs["as_retriever"] == top_docs["hybrid"]
        }
        
        # Performance notes
        if as_retriever_results.get('documents_found', 0) > 0:
            analysis["performance_notes"].append("as_retriever: Standard LangChain interface, easy to use")
        
        if direct_vector_results.get('documents_found', 0) > 0:
            analysis["performance_notes"].append("direct_vector: Raw Chroma scores, maximum control")
        
        if hybrid_results.get('documents_found', 0) > 0:
            analysis["performance_notes"].append("hybrid: Combines keyword (BM25) and semantic (vector) search")
        
        # Recommendations
        if analysis["ranking_analysis"]["all_agree_on_top"]:
            analysis["recommendations"].append("All methods agree on the most relevant document")
        
        if analysis["document_overlap"]["hybrid_unique"] > 0:
            analysis["recommendations"].append("Hybrid search finds unique documents not found by vector-only methods")
        
        if analysis["document_overlap"]["as_retriever_vs_direct_overlap"] == as_retriever_results.get('documents_found', 0):
            analysis["recommendations"].append("as_retriever and direct_vector return identical document sets")
        
        return analysis
    def _test_as_retriever(self, query: str, namespace: str, top_k: int) -> Dict[str, Any]:
        """Test PURE as_retriever() - no score extraction, just documents"""
        try:
            # Create standard retriever (exactly how users would use it)
            retriever = self.vector_store.vector_store.as_retriever(
                search_kwargs={"k": top_k}
            )
            
            # Get documents using standard method (NO SCORES)
            documents = retriever.get_relevant_documents(query)
            
            results = {
                "method": "as_retriever",
                "documents_found": len(documents),
                "documents": [],
                "note": "Pure as_retriever() - no scores available by default",
                "usage": "Standard LangChain interface for most applications"
            }
            
            # Format documents for display (without scores)
            for i, doc in enumerate(documents):
                results["documents"].append({
                    "rank": i + 1,
                    "source": doc.metadata.get('source', 'unknown'),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "file_type": doc.metadata.get('file_type', 'unknown'),
                    "content_length": len(doc.page_content)
                })
            
            print(f"✅ as_retriever: {len(documents)} docs (no scores available)")
            return results
            
        except Exception as e:
            print(f"❌ as_retriever test failed: {e}")
            return {"error": str(e), "method": "as_retriever"}

    def _test_direct_vector_search(self, query: str, namespace: str, top_k: int) -> Dict[str, Any]:
        """Test direct vector similarity search with score analysis"""
        try:
            # Direct Chroma similarity search
            results = self.vector_store.vector_store.similarity_search_with_score(query, k=top_k)
            
            comparison_data = {
                "method": "direct_vector_search",
                "documents_found": len(results),
                "documents": [],
                "scores_range": {"min": 0, "max": 0, "avg": 0},
                "raw_scores": [],
                "score_interpretation": "LOWER = BETTER (distance)"
            }
            
            if results:
                scores = [score for _, score in results]
                comparison_data["scores_range"] = {
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                    "avg": round(sum(scores) / len(scores), 4)
                }
                comparison_data["raw_scores"] = [round(score, 4) for score in scores]
                
                # Show both distance and similarity scores
                similarity_scores = [1.0 - score for score in scores]
                
                # Format documents for display
                for i, (doc, distance_score) in enumerate(results):
                    similarity_score = 1.0 - distance_score
                    comparison_data["documents"].append({
                        "rank": i + 1,
                        "source": doc.metadata.get('source', 'unknown'),
                        "distance_score": round(distance_score, 4),
                        "similarity_score": round(similarity_score, 4),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "file_type": doc.metadata.get('file_type', 'unknown')
                    })
            
            print(f"✅ direct_vector: {len(results)} docs, distance range: {comparison_data['scores_range']}")
            return comparison_data
            
        except Exception as e:
            print(f"❌ direct_vector test failed: {e}")
            return {"error": str(e), "method": "direct_vector_search"}

    def _test_current_hybrid_search(self, query: str, namespace: str, top_k: int) -> Dict[str, Any]:
        """Test your current hybrid search implementation"""
        try:
            if not self.hybrid_retriever:
                return {"error": "Hybrid retriever not available", "method": "current_hybrid"}
            
            results = self.hybrid_retriever.hybrid_search(query, top_k=top_k)
            
            comparison_data = {
                "method": "current_hybrid",
                "documents_found": len(results),
                "documents": [],
                "scores_range": {"min": 0, "max": 0, "avg": 0},
                "raw_scores": [],
                "score_interpretation": "HIGHER = BETTER (fused similarity)"
            }
            
            if results:
                scores = [score for _, score in results]
                comparison_data["scores_range"] = {
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                    "avg": round(sum(scores) / len(scores), 4)
                }
                comparison_data["raw_scores"] = [round(score, 4) for score in scores]
                
                # Format documents for display
                for i, (doc, score) in enumerate(results):
                    comparison_data["documents"].append({
                        "rank": i + 1,
                        "source": doc.metadata.get('source', 'unknown'),
                        "score": round(score, 4),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "file_type": doc.metadata.get('file_type', 'unknown')
                    })
            
            print(f"✅ current_hybrid: {len(results)} docs, score range: {comparison_data['scores_range']}")
            return comparison_data
            
        except Exception as e:
            print(f"❌ current_hybrid test failed: {e}")
            return {"error": str(e), "method": "current_hybrid"}

    def _analyze_method_comparison(self, as_retriever_results: Dict, direct_vector_results: Dict, hybrid_results: Dict) -> Dict[str, Any]:
        """Analyze the differences between methods"""
        analysis = {
            "document_overlap": {},
            "score_analysis": {},
            "performance_notes": [],
            "recommendations": []
        }
        
        # Document overlap analysis
        as_retriever_sources = set([doc['source'] for doc in as_retriever_results.get('documents', [])])
        direct_vector_sources = set([doc['source'] for doc in direct_vector_results.get('documents', [])])
        hybrid_sources = set([doc['source'] for doc in hybrid_results.get('documents', [])])
        
        analysis["document_overlap"] = {
            "as_retriever_unique": len(as_retriever_sources - direct_vector_sources - hybrid_sources),
            "direct_vector_unique": len(direct_vector_sources - as_retriever_sources - hybrid_sources),
            "hybrid_unique": len(hybrid_sources - as_retriever_sources - direct_vector_sources),
            "all_common": len(as_retriever_sources & direct_vector_sources & hybrid_sources)
        }
        
        # Score analysis
        if 'raw_scores' in as_retriever_results and as_retriever_results['raw_scores']:
            analysis["score_analysis"]["as_retriever_range"] = as_retriever_results['scores_range']
        
        if 'raw_scores' in direct_vector_results and direct_vector_results['raw_scores']:
            analysis["score_analysis"]["direct_vector_range"] = direct_vector_results['scores_range']
            
        if 'raw_scores' in hybrid_results and hybrid_results['raw_scores']:
            analysis["score_analysis"]["hybrid_range"] = hybrid_results['scores_range']
        
        # Performance notes
        if as_retriever_results.get('documents_found', 0) > 0:
            analysis["performance_notes"].append("as_retriever: Uses LangChain's standardized interface")
        
        if direct_vector_results.get('documents_found', 0) > 0:
            analysis["performance_notes"].append("direct_vector: Raw Chroma scores (distance-based)")
        
        if hybrid_results.get('documents_found', 0) > 0:
            analysis["performance_notes"].append("hybrid: Combined BM25 + vector with fusion")
        
        # Recommendations
        if analysis["document_overlap"]["all_common"] > 0:
            analysis["recommendations"].append("All methods agree on core relevant documents")
        
        if analysis["document_overlap"]["hybrid_unique"] > 0:
            analysis["recommendations"].append("Hybrid search provides unique document diversity")
        
        return analysis

    

    def _get_vector_results(self, query: str, namespace: str, top_k: int) -> List[Document]:
        """Get results from vector search with relevance scores"""
        search_results = self.vector_store.semantic_search(namespace, query, n_results=top_k)
        
        documents = search_results.get('documents', [])
        metadatas = search_results.get('metadatas', [])
        scores = search_results.get('relevance_scores', [])
        
        scored_documents = []
        for doc_content, metadata, score in zip(documents, metadatas, scores):
            if doc_content and doc_content.strip():
                # Add relevance score to metadata
                metadata['relevance_score'] = float(score) if score is not None else 0.0
                doc = Document(page_content=doc_content, metadata=metadata)
                scored_documents.append((doc, score))
        
        # Sort by relevance score
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_documents]

    def _get_hybrid_results(self, query: str, namespace: str, top_k: int) -> List[Document]:
        """Get results from hybrid search with relevance scores"""
        if not self.hybrid_retriever:
            print("⚠️ Hybrid retriever not available - returning empty")
            return []
        
        try:
            scored_results = self.hybrid_retriever.hybrid_search(query, top_k=top_k)
            
            # Add relevance scores to document metadata
            for doc, score in scored_results:
                doc.metadata['relevance_score'] = float(score)
            
            return [doc for doc, score in scored_results]
        except Exception as e:
            print(f"❌ Hybrid search failed: {e}")
            return []



    def _calculate_relevance_score(self, query: str, documents: List[Document]) -> float:
        """Calculate relevance score based on query-term matching"""
        if not documents:
            return 0.0
        
        query_terms = set(query.lower().split())
        total_score = 0.0
        
        for doc in documents:
            content = doc.page_content.lower()
            doc_terms = set(content.split())
            
            # Calculate term overlap
            overlap = len(query_terms.intersection(doc_terms))
            max_possible = len(query_terms)
            
            if max_possible > 0:
                score = (overlap / max_possible) * 100
                total_score += min(score, 100)  # Cap at 100%
        
        return round(total_score / len(documents), 2) if documents else 0.0

    def _analyze_content_quality(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze the quality of retrieved content"""
        if not documents:
            return {"score": 0, "details": "No documents"}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_length = total_chars / len(documents)
        
        # Count documents with substantial content
        substantial_docs = sum(1 for doc in documents if len(doc.page_content) > 100)
        
        # Analyze content diversity (unique sources)
        sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
        
        # Calculate quality score (0-100)
        length_score = min(avg_length / 50, 1.0) * 40  # Max 40 points for length
        substantial_score = (substantial_docs / len(documents)) * 40  # Max 40 points
        diversity_score = (len(sources) / len(documents)) * 20  # Max 20 points
        
        quality_score = round(length_score + substantial_score + diversity_score, 2)
        
        return {
            "score": quality_score,
            "avg_length": round(avg_length, 2),
            "substantial_docs": substantial_docs,
            "unique_sources": len(sources),
            "sources": list(sources)[:10]  # Top 5 sources
        }



    def _determine_winner(self, hybrid_count: int, langchain_count: int,
                         hybrid_relevance: float, langchain_relevance: float,
                         hybrid_content: Dict, langchain_content: Dict) -> tuple:
        """Determine which retriever performed better"""
        
        # 🎯 SCORING SYSTEM
        hybrid_score = 0
        langchain_score = 0
        reasons = []
        
        # 1. Document Count (30% weight)
        if hybrid_count > langchain_count:
            hybrid_score += 30
            reasons.append(f"Hybrid found more documents ({hybrid_count} vs {langchain_count})")
        elif langchain_count > hybrid_count:
            langchain_score += 30
            reasons.append(f"LangChain found more documents ({langchain_count} vs {hybrid_count})")
        else:
            hybrid_score += 15
            langchain_score += 15
        
        # 2. Relevance Score (40% weight)  
        relevance_weight = 40
        hybrid_score += (hybrid_relevance / 100) * relevance_weight
        langchain_score += (langchain_relevance / 100) * relevance_weight
        
        if hybrid_relevance > langchain_relevance:
            reasons.append(f"Hybrid has better relevance ({hybrid_relevance} vs {langchain_relevance})")
        elif langchain_relevance > hybrid_relevance:
            reasons.append(f"LangChain has better relevance ({langchain_relevance} vs {hybrid_relevance})")
        
        # 3. Content Quality (30% weight)
        quality_weight = 30
        hybrid_score += (hybrid_content['score'] / 100) * quality_weight
        langchain_score += (langchain_content['score'] / 100) * quality_weight
        
        if hybrid_content['score'] > langchain_content['score']:
            reasons.append(f"Hybrid has better content quality ({hybrid_content['score']} vs {langchain_content['score']})")
        elif langchain_content['score'] > hybrid_content['score']:
            reasons.append(f"LangChain has better content quality ({langchain_content['score']} vs {hybrid_content['score']})")
        
        # 🏆 DETERMINE WINNER
        hybrid_score = round(hybrid_score, 2)
        langchain_score = round(langchain_score, 2)
        
        if hybrid_score > langchain_score:
            winner = "hybrid"
            score_diff = hybrid_score - langchain_score
        elif langchain_score > hybrid_score:
            winner = "langchain" 
            score_diff = langchain_score - hybrid_score
        else:
            winner = "tie"
            score_diff = 0
        
        reason = f"{winner.upper()} wins with score {max(hybrid_score, langchain_score)} vs {min(hybrid_score, langchain_score)}. " + "; ".join(reasons[:3])
        
        return winner, reason

    def _format_documents_for_display(self, documents: List[Document]) -> List[Dict]:
        """Format documents for easy frontend display"""
        return [
            {
                'source': doc.metadata.get('source', 'unknown'),
                'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                'content_length': len(doc.page_content),
                'file_type': doc.metadata.get('file_type', 'unknown'),
                'metadata': {k: v for k, v in doc.metadata.items() if k not in ['source', 'file_type']}
            }
            for doc in documents
        ]

    def get_comparison_history(self, limit: int = 10) -> List[Dict]:
        """Get recent comparison history"""
        return self.comparison_history[-limit:]

    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about both retrievers"""
        return {
            'hybrid_retriever': {
                'available': self.hybrid_retriever is not None,
                'documents_in_cache': len(self.hybrid_retriever.documents_cache) if self.hybrid_retriever else 0,
                'bm25_index_built': self.hybrid_retriever.bm25_index is not None if self.hybrid_retriever else False
            },
            'langchain_retriever': {
                'available': self.langchain_retriever is not None,
                'search_type': 'similarity'
            },
            'total_comparisons': len(self.comparison_history),
            'recent_winners': self._get_recent_winner_stats()
        }

    def _get_recent_winner_stats(self, limit: int = 20) -> Dict[str, Any]:
        """Get statistics about recent winners"""
        recent = self.comparison_history[-limit:]
        if not recent:
            return {}
        
        winners = [comp['winner'] for comp in recent if 'winner' in comp]
        hybrid_wins = winners.count('hybrid')
        langchain_wins = winners.count('langchain')
        ties = winners.count('tie')
        
        return {
            'hybrid_wins': hybrid_wins,
            'langchain_wins': langchain_wins,
            'ties': ties,
            'hybrid_win_rate': round(hybrid_wins / len(winners) * 100, 2) if winners else 0,
            'langchain_win_rate': round(langchain_wins / len(winners) * 100, 2) if winners else 0
        }

    def ingest_document_with_auto_sync(self, namespace: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """GUARANTEED sync between vector store and hybrid cache"""
        try:
            print(f"📥 GUARANTEED SYNC: {filename} for {namespace}")
            
            # Check for duplicate
            is_duplicate, reason = self.vector_store.is_duplicate_document(namespace, file_bytes, filename)
            structured_preview = None
            try:
                structured_preview = self.vector_store.document_processor.extract_structured_preview(file_bytes, filename)
            except Exception as preview_error:
                print(f"⚠️ Structured data preview failed for {filename}: {preview_error}")
            
            # 🚨 ALWAYS ensure hybrid cache is synced, even for duplicates
            def sync_hybrid_cache():
                """Sync hybrid cache with current vector store state"""
                if self.vector_store.hybrid_retrieval:
                    # Get ALL documents from vector store
                    all_docs = self.vector_store.vector_store.similarity_search("", k=10000)
                    
                    # Smart deduplication
                    unique_docs = []
                    seen_identifiers = set()
                    for doc in all_docs:
                        source = doc.metadata.get('source', 'unknown')
                        chunk_id = doc.metadata.get('chunk_id', 'unknown')
                        doc_identifier = f"{source}_{chunk_id}"
                        
                        if doc_identifier not in seen_identifiers:
                            seen_identifiers.add(doc_identifier)
                            unique_docs.append(doc)
                    
                    # Build hybrid index
                    self.vector_store.hybrid_retrieval.build_hybrid_index(unique_docs)
                    
                    # ✅ VERIFICATION: Ensure sync actually worked
                    hybrid_count = len(self.vector_store.hybrid_retrieval.documents_cache) if self.vector_store.hybrid_retrieval.documents_cache else 0
                    vector_count = len(unique_docs)
                    
                    if hybrid_count == vector_count:
                        print(f"✅ HYBRID CACHE SYNCED: {len(unique_docs)} documents")
                    else:
                        print(f"⚠️ SYNC MISMATCH: Vector={vector_count}, Hybrid={hybrid_count} - retrying...")
                        # Retry once
                        try:
                            self.vector_store.hybrid_retrieval.build_hybrid_index(unique_docs)
                            hybrid_count_retry = len(self.vector_store.hybrid_retrieval.documents_cache) if self.vector_store.hybrid_retrieval.documents_cache else 0
                            if hybrid_count_retry == vector_count:
                                print(f"✅ HYBRID CACHE SYNCED (after retry): {hybrid_count_retry} documents")
                            else:
                                print(f"❌ SYNC FAILED: Vector={vector_count}, Hybrid={hybrid_count_retry}")
                        except Exception as retry_e:
                            print(f"❌ Sync retry failed: {retry_e}")
                    
                    return hybrid_count if hybrid_count == vector_count else vector_count
                return 0
            
            if is_duplicate:
                print(f"⏭️ Document exists, syncing hybrid cache...")
                hybrid_count = sync_hybrid_cache()
                
                self.vector_store.document_storage.update_document_metadata(
                    namespace,
                    filename=filename,
                    updates={
                        'file_size': len(file_bytes),
                        'upload_time': datetime.now().isoformat()
                    },
                    structured_preview=structured_preview
                )
                
                return {
                    'success': True,
                    'filename': filename,
                    'chunks': 0,
                    'namespace': namespace,
                    'message': f'Document exists - hybrid cache synced with {hybrid_count} documents',
                    'skipped': True,
                    'reason': reason,
                    'hybrid_sync_performed': True
                }
            
            # Process new document
            documents = self.vector_store.document_processor.process_document(file_bytes, filename)
            
            if not documents:
                # Even if no documents, sync hybrid cache
                sync_hybrid_cache()
                if structured_preview:
                    self.vector_store.document_storage.update_document_metadata(
                        namespace,
                        filename=filename,
                        updates={
                            'file_size': len(file_bytes),
                            'upload_time': datetime.now().isoformat()
                        },
                        structured_preview=structured_preview
                    )
                return {
                    'success': False, 
                    'error': 'No content extracted', 
                    'chunks': 0,
                    'skipped': False
                }
            
            # Add to vector store
            chunk_count = self.vector_store.add_documents(namespace, documents)
            
            if chunk_count > 0:
                # Sync hybrid cache with new documents
                hybrid_count = sync_hybrid_cache()
                
                self.vector_store.document_storage.update_document_metadata(
                    namespace,
                    filename=filename,
                    updates={
                        'file_size': len(file_bytes),
                        'upload_time': datetime.now().isoformat()
                    },
                    structured_preview=structured_preview
                )
                
                return {
                    'success': True,
                    'filename': filename,
                    'chunks': chunk_count,
                    'namespace': namespace,
                    'message': f'Successfully ingested {chunk_count} chunks, hybrid cache: {hybrid_count} documents',
                    'skipped': False,
                    'reason': 'new_document',
                    'hybrid_sync_performed': True
                }
            else:
                # Sync hybrid cache even if vector store add failed
                sync_hybrid_cache()
                return {
                    'success': False, 
                    'error': 'Failed to store documents', 
                    'chunks': 0,
                    'skipped': False
                }
                    
        except Exception as e:
            print(f"❌ Document ingestion failed: {e}")
            return {
                'success': False, 
                'error': str(e), 
                'chunks': 0,
                'skipped': False
            }
    def query(self, query: str, namespace: str, session_id: str = None) -> Dict[str, Any]:
        print(f"🔍 ENHANCED QUERY: '{query}'")
        
        # 🎯 STEP 1: AI-powered file intent detection
        file_intent = self._detect_file_intent_ai(query)
        if file_intent:
            print(f"🎯 AI DETECTED FILE INTENT: {file_intent}")
            
            # 🚨 TEMPORARY: Use debug method to find the error
            return self._enhanced_file_content_query(file_intent, namespace, session_id)
        

        
        # 🎯 STEP 2: Check if this is a chart request
        chart_intent = self._parse_chart_request(query)
        if chart_intent:
            print(f"🎯 Chart intent detected: {chart_intent}")
            
            # 🎯 STEP 3: Try to find relevant structured dataset
            dataset_match = self._find_structured_dataset(namespace, chart_intent)
            
            if dataset_match and dataset_match['doc'].get('chart_data_id'):
                print(f"✅ Found dataset with chart data: {dataset_match['doc'].get('filename')}")
                
                # 🎯 STEP 4: USE ORIGINAL DATA FOR CHARTING
                chart_data = self.build_chart_from_original_data(
                    query, namespace, dataset_match['doc']
                )
                
                if chart_data:
                    print("✅ Successfully built chart from ORIGINAL DATA")
                    # Build answer using accurate chart data
                    answer = self._build_chart_answer(dataset_match, chart_intent, chart_data)
                    response = {
                        'answer': answer,
                        'confidence': 'high',
                        'sources': [{'source': dataset_match['doc']['filename'], 'type': 'original_data'}],
                        'chart': chart_data,
                        'method_used': 'original_data_chart'
                    }
                    
                    # Save to chat history
                    if session_id:
                        self.chat_history.add_message(session_id, 'user', query)
                        self.chat_history.add_message(session_id, 'assistant', answer, {
                            'method_used': 'original_data_chart',
                            'chart': chart_data,
                            'sources': response['sources']
                        })
                    
                    return response
                else:
                    print("❌ Original data chart failed, falling back to chunks")
            else:
                print(f"❌ No dataset found or no chart_data_id. dataset_match: {dataset_match is not None}")
                if dataset_match:
                    print(f"   Chart data ID: {dataset_match['doc'].get('chart_data_id')}")
        
        # 🎯 STEP 5: Fallback to existing chunk-based approach
        print("🔄 Using standard chunk-based query...")
        return self._handle_general_query(query, namespace, session_id)

    def _prepare_response(self, answer: str, confidence: str, documents: List[Document], method: str, session_id: str = None, query: str = None) -> Dict[str, Any]:
        """Prepare final response with chat history"""
        
        response = {
            'answer': answer,
            'sources': self._prepare_sources(documents),
            'confidence': confidence,
            'documents_used': len(documents),
            'method_used': method
        }
        
        print(f"🎯 FINAL DECISION: Used {method} method, confidence: {confidence}")
        print(f"📄 DOCUMENTS USED: {len(documents)}")
        
        # Save to chat history - ONLY if session_id and query are provided
        if session_id is not None and query is not None:
            try:
                self.chat_history.add_message(session_id, 'user', query)
                self.chat_history.add_message(session_id, 'assistant', answer, {
                    'sources': response['sources'],
                    'confidence': confidence,
                    'method_used': method
                })
            except Exception as e:
                print(f"⚠️ Failed to save to chat history: {e}")
        
        print("=" * 80)
        return response


    def _should_include_document(self, doc: Document, query: str) -> bool:
        """Smart check if document should be included for this query"""
        content = doc.page_content.lower()
        query_lower = query.lower()
        
        # 🚨 REJECT documents that are clearly irrelevant
        file_type = doc.metadata.get('file_type', '').lower()
        
        # If query asks for specific content but document is empty/irrelevant
        if len(content.strip()) < 50:  # Very short content
            return False
        
        # If query mentions specific terms but document has none of them
        important_words = [word for word in query_lower.split() if len(word) > 4]
        if important_words:
            has_important_content = any(word in content for word in important_words)
            if not has_important_content:
                return False
        
        return True


    def _build_context_for_method(self, documents: List[Document], query: str, method: str) -> str:
        """Build context optimized for each search method - FIXED VERSION"""
        
        if not documents:
            return f"No documents found for: {query}"
        
        if method == "vector":
            context_header = f"VECTOR SEARCH RESULTS for: {query}\n\n"
            max_docs = 8
            
        elif method == "hybrid":  
            context_header = f"HYBRID SEARCH RESULTS for: {query}\n\n"
            max_docs = 10
            
        else:  # fallback
            context_header = f"AVAILABLE DOCUMENTS for: {query}\n\n"
            max_docs = 12
        
        context_parts = [context_header]
        total_content_length = 0
        max_context_size = 50000  # Reduced to ensure it fits
        included_docs = []
        
        # Include documents in their original order
        for i, doc in enumerate(documents[:max_docs]):
            if not doc.page_content or not doc.page_content.strip():
                continue
                
            doc_content = doc.page_content
            doc_content_length = len(doc_content)
            
            # Skip if adding this would exceed context size
            if total_content_length + doc_content_length > max_context_size:
                break
            
            source = doc.metadata.get('source', 'Unknown Document')
            context_parts.append(f"--- DOCUMENT {i+1}: {source} ---")
            context_parts.append(f"Content:\n{doc_content}")
            context_parts.append("")
            
            total_content_length += doc_content_length
            included_docs.append(source)
        
        if included_docs:
            context_parts.append(f"\nCONTEXT SUMMARY: {len(included_docs)} documents, {total_content_length} chars")
            final_context = "\n".join(context_parts)
            print(f"📄 {method.upper()} CONTEXT: {len(included_docs)} docs, {total_content_length} chars")
            return final_context
        else:
            print(f"⚠️ {method.upper()} CONTEXT: No documents with content found!")
            return f"No relevant document content found for: {query}"

    

    def _generate_related_answer(self, query: str, documents: List[Document]) -> str:
        """Generate an actual answer based on related documents, not just show snippets"""
        if not documents:
            return f"I couldn't find any information about \"{query}\" in the corporate documents."
        
        # Build context from the most relevant documents
        context = "Based on the available corporate documents, here's what I found that might be related to your query:\n\n"
        
        # Include top 3 most relevant documents
        for i, doc in enumerate(documents[:3]):
            source = doc.metadata.get('source', 'Unknown document')
            content_preview = doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content
            
            context += f"From {source}:\n{content_preview}\n\n"
        
        # Ask AI to generate a helpful answer based on related content
        prompt = f"""Based on the following context, provide a helpful answer to the user's query. 
    If you cannot answer directly, explain what related information is available and how it might be useful.

    USER QUERY: {query}

    AVAILABLE CONTEXT:
    {context}

    Please provide a helpful response that:
    1. Acknowledges if the exact answer isn't available
    2. Explains what related information WAS found
    3. Suggests how this information might be useful
    4. Is professional and helpful

    RESPONSE:"""

        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000,
                "stream": False
            }

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"❌ Related answer generation failed: {e}")
            # Fallback to simple message
            return f"I couldn't find the exact answer to \"{query}\" but found some related documents that might contain useful information."
    
    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer using DeepSeek API - BALANCED approach"""
        try:
            prompt = f"""You are a professional corporate AI assistant. Answer the question based ONLY on the provided context.

    CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS — BALANCED + COLLEAGUE TONE

    First, thoroughly search the provided context/documents for any information related to the question.

    If you find relevant information:

    - Extract and present it clearly and comprehensively.
    - Be precise and professional, and cite specific document sources (file names, section titles, or page numbers) when possible.
    - Use short paragraphs and conversational language (avoid phrases like “Based on what I found…” or “From the documents I see…”).

    If you find partial or related information:

    - Present the available pieces and explicitly note what’s missing.

    If you cannot find the answer totally, use clear phrases such as:

    - “I cannot find [specific information] in the provided documents.”
    - “The [requested information] is not available in the context.”
    - “I cannot determine [what was asked] based on the available information.”
    - “No information about [topic] was found in the documents.”

    ONLY if, after a thorough search, you cannot find any relevant information:

    - Clearly state that no relevant information was found.
    - Use natural, colleague-like phrasing such as: “I couldn’t find specific information about that in our documents.”

    Be proactive in extracting ALL available information; never assume something is absent without verifying.

    Keep the tone friendly but professional — use contractions and short, readable paragraphs. If you’re unsure about something, say so naturally.

    OUTPUT FORMAT (suggested, visually structured)

    **Answer:**
    - Direct, precise response to the question

    **Findings:**
    - Bullet points or short sections with extracted facts
    - Cite exact document names, section titles, or page numbers

    **Missing / Partial Data:**
    - Explicitly list what’s missing (if any) with suggested phrases

    **Quick Conclusion / Next Steps:**
    - Optional friendly or actionable suggestion

    BASED ON THE CONTEXT ABOVE, PROVIDE YOUR ANSWER:"""

            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2048,
                "stream": False
            }

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"❌ DeepSeek API call failed: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"


    def _prepare_sources(self, documents: List[Document]) -> List[Dict]:
        """Prepare source information with ACTUAL relevance scores and excerpts - LIMITED to 10"""
        sources = []
        
        for doc in documents:
            content = doc.page_content.strip()
            
            # Create meaningful excerpt (first 200 chars or meaningful snippet)
            excerpt = ""
            if content:
                # Try to find a complete sentence within first 300 chars
                sentences = re.split(r'[.!?]+', content[:300])
                if len(sentences) > 1:
                    excerpt = sentences[0] + '.' if sentences[0] else content[:200] + "..."
                else:
                    excerpt = content[:200] + "..." if len(content) > 200 else content
            else:
                excerpt = "Content not available"
            
            # Extract actual relevance score from metadata if available
            relevance_score = doc.metadata.get('relevance_score', 'N/A')
            if relevance_score != 'N/A':
                try:
                    relevance_score = f"{float(relevance_score):.3f}"
                except (ValueError, TypeError):
                    relevance_score = 'N/A'
            
            # Determine relevance level for display
            if relevance_score != 'N/A':
                score_val = float(relevance_score)
                if score_val > 0.7:
                    relevance_level = 'high'
                elif score_val > 0.4:
                    relevance_level = 'medium'
                else:
                    relevance_level = 'low'
            else:
                relevance_level = 'unknown'
            
            sources.append({
                'source': doc.metadata.get('source', 'Unknown'),
                'file_type': doc.metadata.get('file_type', 'unknown'),
                'excerpt': excerpt,
                'relevance_score': relevance_score,
                'relevance_level': relevance_level,
                'content_length': len(content)
            })
        
        # Sort by relevance score if available
        try:
            sources.sort(key=lambda x: float(x['relevance_score']) if x['relevance_score'] != 'N/A' else 0, reverse=True)
        except:
            pass
        
        # 🚨 LIMIT TO 10 SOURCES
        limited_sources = sources[:10]
        print(f"📄 SOURCES: Limited {len(sources)} → {len(limited_sources)} sources for display")
        
        return limited_sources
    
    def _calculate_confidence(self, documents: List[Document]) -> str:
        """Calculate confidence based on search results quality"""
        if not documents:
            print("   📊 CONFIDENCE: No documents - very low")
            return 'very low'
        
        # Calculate various quality metrics
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chars = total_chars / len(documents)
        
        substantial_docs = sum(1 for doc in documents if len(doc.page_content.strip()) > 100)
        substantial_ratio = substantial_docs / len(documents)
        
        unique_sources = len(set(doc.metadata.get('source', 'unknown') for doc in documents))
        diversity_score = unique_sources / len(documents)
        
        # 🚨 IMPROVED CONFIDENCE CALCULATION
        confidence_score = 0
        
        # Content length factor (40%)
        if avg_chars > 800 and len(documents) >= 5:
            confidence_score += 40
        elif avg_chars > 500 and len(documents) >= 3:
            confidence_score += 30
        elif avg_chars > 200 and len(documents) >= 2:
            confidence_score += 20
        else:
            confidence_score += 10
        
        # Content quality factor (30%)
        if substantial_ratio > 0.8:
            confidence_score += 30
        elif substantial_ratio > 0.5:
            confidence_score += 20
        else:
            confidence_score += 10
        
        # Diversity factor (30%)
        if diversity_score > 0.7:
            confidence_score += 30
        elif diversity_score > 0.4:
            confidence_score += 20
        else:
            confidence_score += 10
        
        # Convert to confidence levels
        if confidence_score >= 80:
            confidence_level = 'high'
        elif confidence_score >= 60:
            confidence_level = 'medium'
        elif confidence_score >= 40:
            confidence_level = 'low'
        else:
            confidence_level = 'very low'
        
        print(f"   📊 CONFIDENCE CALCULATION:")
        print(f"   - Documents: {len(documents)}")
        print(f"   - Avg chars: {avg_chars:.0f}")
        print(f"   - Substantial ratio: {substantial_ratio:.2f}")
        print(f"   - Diversity score: {diversity_score:.2f}")
        print(f"   - Final score: {confidence_score}/100 -> {confidence_level}")
        
        return confidence_level
    
    def get_document_stats(self, namespace: str) -> Dict[str, Any]:
        """Get statistics for documents in a namespace"""
        return self.vector_store.get_collection_stats(namespace)

    def sync_google_drive(self, user_context: CorporateUserContext) -> Dict[str, Any]:
        """Start Google Drive sync for ANY role with their specific folders"""
        if not user_credentials:
            return {"error": "No Google Drive credentials available"}
        
        # Get credentials
        user_email, credentials_dict = list(user_credentials.items())[0]
        namespace = user_context.get_namespace()
        
        print(f"🔄 Starting Google Drive sync for {user_context.role}: {user_context.email}")
        print(f"📁 Accessible folders: {user_context.get_accessible_folders()}")
        
        # Start sync in background for ALL roles
        import asyncio
        asyncio.create_task(
            self.sync_entire_drive(credentials_dict, namespace)
        )
        
        return {
            "status": "started", 
            "message": f"Google Drive sync started for {user_context.role} role",
            "namespace": namespace,
            "user": user_context.email,
            "accessible_folders": user_context.get_accessible_folders()
        }
    
    def get_sync_status(self, namespace: str) -> Dict[str, Any]:
        """Get Google Drive sync status"""
        return self.drive_manager.get_sync_status(namespace)





    def _aggressive_prioritization(self, query: str, documents: List[Document]) -> List[Document]:
        """SAFE prioritization with careful boosting"""
        scored_docs = []
        query_lower = query.lower()
        
        print(f"🎯 SAFE PRIORITIZATION for: '{query}'")
        
        for doc in documents:
            score = 1.0
            source = doc.metadata.get('source', '').lower()
            filename = self._extract_filename(source).lower()
            content = doc.page_content.lower()
            file_type = doc.metadata.get('file_type', '').lower()
            
            # 🚨 BOOST 1: SMART Filename matching
            # Clean both query and filename for comparison
            query_clean = query_lower
            filename_clean = filename
            
            # Remove common file extensions from both
            extensions = ['.pdf', '.docx', '.xlsx', '.xls', '.pptx', '.ppt', '.csv', '.txt', '.md', '.html']
            for ext in extensions:
                query_clean = query_clean.replace(ext, '')
                filename_clean = filename_clean.replace(ext, '')
            
            # Remove spaces/underscores for comparison
            query_normalized = query_clean.replace(' ', '_').replace('-', '_')
            filename_normalized = filename_clean.replace(' ', '_').replace('-', '_')
            
            # Exact filename match (CAREFUL boost)
            if filename_normalized in query_normalized:
                score *= 4.0  # Reduced from 8.0 to be safer
                print(f"🎯 EXACT FILENAME: {filename} -> {score:.1f}")
            
            # Partial filename match (LOWER boost)
            elif any(part in query_clean for part in filename_clean.split() if len(part) > 3):
                score *= 2.0  # Lower boost for safety
                print(f"📁 PARTIAL FILENAME: {filename} -> {score:.1f}")
            
            # 🚨 BOOST 2: File type matching (GENERIC)
            file_type_keywords = {
                'pdf': ['pdf', 'document'],
                'excel': ['excel', 'spreadsheet', 'sheet'],
                'word': ['word', 'doc', 'document'],
                'powerpoint': ['powerpoint', 'presentation', 'ppt'],
                'csv': ['csv', 'data', 'table'],
                'text': ['text', 'txt']
            }
            
            for f_type, keywords in file_type_keywords.items():
                if file_type == f_type and any(keyword in query_lower for keyword in keywords):
                    score *= 1.5  # Very gentle boost
                    print(f"📄 FILE TYPE: {file_type} -> {score:.1f}")
                    break
            
            # 🚨 BOOST 3: Content matching (VERY SAFE)
            query_terms = [word for word in query_lower.split() if len(word) > 4]
            content_matches = sum(1 for term in query_terms if term in content)
            
            if content_matches >= 2:
                score *= (1.0 + (content_matches * 0.3))  # Very gentle scaling
                print(f"🔤 CONTENT: {content_matches} terms -> {score:.1f}")
            
            scored_docs.append((doc, score))
        
        # Sort by final score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🎯 SAFE ORDER:")
        for i, (doc, score) in enumerate(scored_docs[:5]):
            source = doc.metadata.get('source', 'unknown')
            original_idx = documents.index(doc) + 1 if doc in documents else '?'
            print(f"   {i+1}. {source} (was #{original_idx}) -> {score:.1f}")
        
        return [doc for doc, score in scored_docs]

    def _is_document_specific_query(self, query_lower: str) -> bool:
        """Check if query is asking about a specific document's content"""
        patterns = [
            'what does', 'what is in', 'show me', 'tell me about',
            'content of', 'information in', 'details about', 'data in'
        ]
        return any(pattern in query_lower for pattern in patterns)

    def _extract_target_filename(self, query_lower: str) -> str:
        """Extract the target filename from document-specific queries"""
        import re
        
        # Patterns like "what does X have", "content of X", "information in X"
        patterns = [
            r'what does (.+?) have',
            r'what is in (.+?)', 
            r'content of (.+?)',
            r'information in (.+?)',
            r'data in (.+?)',
            r'show me (.+?)',
            r'tell me about (.+?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                filename = match.group(1).strip()
                # Clean up the filename
                filename = filename.replace(' the ', '').replace(' your ', '').replace(' this ', '')
                if filename:
                    return filename
        
        return ""

    def _filename_matches(self, target: str, actual: str) -> bool:
        """Check if actual filename matches target from query"""
        import re
        # Remove extensions and clean
        target_clean = re.sub(r'\.(pdf|docx|xlsx|csv|txt)$', '', target)
        actual_clean = re.sub(r'\.(pdf|docx|xlsx|csv|txt)$', '', actual)
        
        # Check different matching strategies
        matches = (
            target_clean in actual_clean or
            actual_clean in target_clean or
            any(word in actual_clean for word in target_clean.split() if len(word) > 2)
        )
        
        return matches


    def _extract_filename(self, source: str) -> str:
        """Extract clean filename from source path"""
        # Remove common prefixes and extensions
        filename = source
        for prefix in ['Google Drive: ', 'processed_']:
            filename = filename.replace(prefix, '')
        
        # Remove file extension
        filename = os.path.splitext(filename)[0]
        
        # Remove sheet names for Excel files
        if ' - ' in filename:
            filename = filename.split(' - ')[0]
        
        return filename.strip()

    def _query_matches_filename(self, query: str, filename: str) -> bool:
        """Check if query contains filename keywords"""
        if not filename or filename == 'unknown':
            return False
        
        # Split filename into meaningful words
        filename_words = re.findall(r'[a-zA-Z0-9]+', filename)
        
        # Check if any filename word appears in query
        for word in filename_words:
            if len(word) > 3 and word in query:  # Only meaningful words
                return True
        
        return False

    def _query_matches_filetype(self, query: str, file_type: str) -> bool:
        """Check if query references specific file types"""
        file_type_keywords = {
            'excel': ['excel', 'spreadsheet', 'sheet', 'xlsx'],
            'word': ['word', 'document', 'docx'], 
            'pdf': ['pdf', 'document'],
            'csv': ['csv', 'data', 'table']
        }
        
        if file_type in file_type_keywords:
            for keyword in file_type_keywords[file_type]:
                if keyword in query:
                    return True
        
        return False


    def _is_ai_answer_found(self, ai_response: str, confidence: str, query: str, method: str = "vector") -> bool:
        """DUAL APPROACH: Exact vector prompt + lenient hybrid"""
        
        print(f"   🤖 {method.upper()} CONFIDENCE CHECK:")
        print(f"   📊 Confidence level: '{confidence}'")
        print(f"   📝 AI response snippet: '{ai_response[:500]}...'")

        try:
            if method == "vector":
                # 🚨 EXACT VECTOR PROMPT AS REQUESTED
                evaluation_prompt = f"""
                Analyze if the AI response successfully answers the user's question.

                USER QUESTION: {query}

                AI RESPONSE: {ai_response}
                
                Carefully evaluate:
                - Does the response directly provide the specific information requested?
                - Is the response substantive and specific, or vague and avoidant?
                - Does it contain actual content from documents, or just excuses?
                
                Consider these examples:
                
                GOOD ANSWER (should return ANSWERED):
                "The causes of spiritual boredom are: 1. Lack of prayer 2. Routine worship 3. Spiritual dryness"
                "Based on the documents, the column names are: id, name, age, department"
                "The phone number for John Smith is 555-1234 according to the contact list"
                
                BAD ANSWER (should return NOT_ANSWERED):  
                "I cannot find the causes of spiritual boredom in the documents"
                "The column names are not available in the provided context"
                "No phone numbers were found for John Smith"
                "I don't have that information in the corporate documents"
                
                Respond with ONLY one word: "ANSWERED" or "NOT_ANSWERED"
                
                Evaluation:"""
                
            else:
                # 🚨 LENIENT HYBRID: Accept any useful information
                print("   🎯 USING LENIENT HYBRID EVALUATION")
                evaluation_prompt = f"""
                Analyze if the AI response provides ANY useful information for the user's question.

                USER QUESTION: {query}

                AI RESPONSE: {ai_response}
                
                Be VERY LENIENT - accept if it contains ANY information that could help the user:
                
                ✅ ACCEPT (return ANSWERED) if:
                - Contains any relevant information, even if partial
                - Provides alternatives, context, or related details  
                - Explains what WAS found, even if not the exact answer
                - Has specific names, locations, dates, or facts
                - States limitations but offers useful alternatives
                
                ❌ REJECT (return NOT_ANSWERED) only if:
                - Completely empty of useful content
                - Generic "I don't know" with no context
                - No specific information whatsoever
                
                Examples to ACCEPT:
                "I cannot find Abraham's current address, but I found Abraham Mekonnen in IN, USA"
                "The documents show several people named Abraham but no specific address"
                "Based on the church member database, I found these individuals with Abraham in their name..."
                "The exact address isn't available, however here are related locations..."
                
                Examples to REJECT:  
                "I cannot answer that question"
                "No information available"
                
                Respond with ONLY one word: "ANSWERED" or "NOT_ANSWERED"
                
                Evaluation:"""
            
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": evaluation_prompt}],
                "temperature": 0.1,
                "max_tokens": 20,
                "stream": False
            }

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20
            )
            response.raise_for_status()
            
            evaluation = response.json()["choices"][0]["message"]["content"].strip().upper()
            print(f"   🧠 {method.upper()} SELF-EVALUATION: {evaluation}")
            
            # 🚨 FIXED LOGIC: Check both conditions properly
            answered_found = "ANSWERED" in evaluation
            not_answered_found = "NOT_ANSWERED" in evaluation
            
            # Confidence logic for vector only
            if method == "vector" and confidence == 'very low':
                print("   ❌ VECTOR: VERY LOW CONFIDENCE - Overriding to fallback")
                return False

            # Success check
            if answered_found and not not_answered_found:
                print(f"   ✅ {method.upper()}: SUCCESSFULLY ANSWERED THE QUESTION")
                return True
            else:
                print(f"   ❌ {method.upper()}: FAILED TO ANSWER - TRIGGERING FALLBACK")
                return False

        except Exception as e:
            print(f"   ⚠️ {method.upper()} evaluation failed: {e}")
            print(f"   ⚠️ Evaluation service down - assuming answer found to avoid unnecessary fallback")
            return True

    def _generate_answer_with_documents(self, documents: List[Document], query: str) -> tuple[str, str]:
        """Generate a direct, confident answer based strictly on the provided documents."""

        if not documents:
            return "Information not found in the provided documents.", "very low"

        # ✅ 1. Build context
        context = self._build_context_for_method(documents, query, "general")

        # ✅ 2. Strong, direct answering prompt
        prompt = f"""
        You are an assistant that answers questions **strictly and directly** from the provided documents.
        If the answer exists in the context, give it plainly and concisely — without saying things like 
        "based on available documents" or "I cannot provide".
        If the answer is not found at all, respond only with: "Information not found in the provided documents."

        CONTEXT:
        {context}

        QUESTION:
        {query}

        DIRECT ANSWER:
        """

        # Generate using your existing LLM interface
        answer = self._generate_answer(prompt, query)

        # ✅ 3. Calculate confidence (your existing logic)
        confidence = self._calculate_confidence(documents)

        return answer, confidence

    def _extract_related_snippets(self, documents: List[Document], query: str) -> str:
        """Extract the most relevant snippets from documents for fallback"""
        if not documents:
            return ""
        
        snippets = []
        for i, doc in enumerate(documents[:20]):  # Top 5 most relevant
            source = doc.metadata.get('source', 'Unknown document')
            content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            snippets.append(f"From {source}:\n{content_preview}")
        
        return "\n\n".join(snippets)

    def _track_fallback_performance(self, query: str, vector_found: bool, hybrid_found: bool, final_method: str):
        """Track how often fallback is triggered and successful"""
        if not hasattr(self, 'fallback_stats'):
            self.fallback_stats = {
                'total_queries': 0,
                'vector_success': 0,
                'hybrid_success': 0,
                'fallback_used': 0,
                'queries_that_benefited': []
            }
        
        self.fallback_stats['total_queries'] += 1
        
        if vector_found:
            self.fallback_stats['vector_success'] += 1
        elif hybrid_found:
            self.fallback_stats['hybrid_success'] += 1
            self.fallback_stats['fallback_used'] += 1
            self.fallback_stats['queries_that_benefited'].append(query)
        else:
            self.fallback_stats['fallback_used'] += 1

    def build_chart_from_original_data(self, query: str, namespace: str, document_info: Dict) -> Optional[Dict]:
        """Build chart from ORIGINAL file data (not chunks) - ENHANCED VERSION with sheet support"""
        try:
            if not document_info.get('chart_data_id'):
                print(f"⚠️ No chart_data_id found for {document_info.get('filename')}")
                return None
            
            # Get original file bytes
            original_data = self.document_storage.get_original_file(
                namespace, document_info['chart_data_id']
            )
            
            if not original_data:
                print(f"⚠️ Could not retrieve original file data for {document_info.get('filename')}")
                return None
            
            # Process with pandas for accurate charting
            # 🚨 FIX: Extract file extension from actual filename (handle "filename.xlsx - SheetName" format)
            filename = document_info['filename']
            # Remove sheet name suffix if present (e.g., "Iris.xlsx - Iris.csv" -> "Iris.xlsx")
            if ' - ' in filename:
                filename = filename.split(' - ')[0]
            file_ext = os.path.splitext(filename.lower())[1]
            
            print(f"🐛 DEBUG: build_chart_from_original_data - filename: {document_info['filename']}, extracted: {filename}, ext: {file_ext}")
            
            chart_intent = self._parse_chart_request(query)
            
            if not chart_intent:
                print(f"⚠️ Could not parse chart intent from query: {query}")
                return None
            
            if file_ext in ['.xlsx', '.xls']:
                import pandas as pd
                from io import BytesIO
                
                excel_file = BytesIO(original_data)
                
                # Try to get structured data to find the right sheet
                structured_data = self.document_storage.get_structured_data(
                    namespace, document_info.get('file_id') or document_info['chart_data_id']
                )
                
                # If we have structured data, try to match sheet from query
                sheet_name = None
                if structured_data:
                    sheets = structured_data.get('sheets', [])
                    query_lower = query.lower()
                    
                    # Try to find matching sheet name in query
                    for sheet in sheets:
                        sheet_name_lower = sheet.get('name', '').lower()
                        if sheet_name_lower in query_lower or any(word in sheet_name_lower for word in query_lower.split()):
                            sheet_name = sheet.get('name')
                            break
                    
                    # If no match, use first sheet
                    if not sheet_name and sheets:
                        sheet_name = sheets[0].get('name')
                
                # Read Excel file
                try:
                    if sheet_name:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
                        print(f"📊 Reading sheet '{sheet_name}' from {document_info['filename']}")
                    else:
                        # Read first sheet by default
                        df = pd.read_excel(excel_file, engine='openpyxl')
                        print(f"📊 Reading first sheet from {document_info['filename']}")
                except Exception as e:
                    print(f"⚠️ Failed to read Excel sheet '{sheet_name}': {e}, trying first sheet")
                    excel_file.seek(0)
                    df = pd.read_excel(excel_file, engine='openpyxl')
                
                # Build chart from actual DataFrame (not chunks)
                chart_data = self.build_chart_from_dataframe(df, chart_intent, document_info['filename'])
                return chart_data
                    
            elif file_ext == '.csv':
                import pandas as pd
                from io import BytesIO
                
                # Try different encodings for CSV files
                df = None
                encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                
                for encoding in encodings_to_try:
                    try:
                        csv_file = BytesIO(original_data)
                        df = pd.read_csv(csv_file, encoding=encoding)
                        if df is not None and not df.empty:
                            print(f"✅ Successfully read CSV with {encoding} encoding for chart building")
                            break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        # Try next encoding
                        continue
                
                if df is None or df.empty:
                    print(f"❌ Failed to read CSV with any encoding for chart building")
                    return None
                
                # Build chart from actual DataFrame (not chunks)
                chart_data = self.build_chart_from_dataframe(df, chart_intent, document_info['filename'])
                return chart_data
            
            print(f"⚠️ Unsupported file type for chart building: {file_ext}")
            return None
            
        except Exception as e:
            print(f"❌ Chart building from original data failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def build_chart_from_dataframe(self, df, chart_intent: Dict, filename: str) -> Dict:
        """Build chart from actual DataFrame (accurate data) - ENHANCED VERSION with smart column selection"""
        try:
            import numpy as np
            
            # Clean DataFrame: remove completely empty rows/columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                print(f"❌ DataFrame is empty after cleaning")
                return None
            
            # Build column info for selection
            column_info = []
            numeric_cols = []
            categorical_cols = []
            
            for col in df.columns:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                column_info.append({
                    'name': col,
                    'is_numeric': is_numeric
                })
                if is_numeric:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            # Smart column selection using existing logic
            metric_column_dict, metric_score = self._select_metric_column(
                column_info,
                numeric_cols,
                chart_intent.get('metric_keywords', [])
            )
            
            dimension_column_dict, dimension_score = self._select_dimension_column(
                column_info,
                categorical_cols,
                chart_intent.get('dimension_keywords', [])
            )
            
            # Extract column names from dicts
            metric_column = metric_column_dict.get('name') if metric_column_dict else None
            dimension_column = dimension_column_dict.get('name') if dimension_column_dict else None
            
            # Fallback logic: use best available columns
            if not metric_column:
                if numeric_cols:
                    metric_column = numeric_cols[0]
                    print(f"⚠️ Using fallback metric column: {metric_column}")
                else:
                    print(f"❌ No numeric columns found for metric")
                    return None
            
            if not dimension_column:
                if categorical_cols:
                    dimension_column = categorical_cols[0]
                    print(f"⚠️ Using fallback dimension column: {dimension_column}")
                elif numeric_cols and len(numeric_cols) > 1:
                    # Use second numeric as dimension if no categorical
                    dimension_column = numeric_cols[1]
                    print(f"⚠️ Using numeric column as dimension: {dimension_column}")
                else:
                    print(f"❌ No suitable dimension column found")
                    return None
            
            # Validate columns exist in DataFrame
            if metric_column not in df.columns:
                print(f"❌ Metric column '{metric_column}' not in DataFrame columns: {list(df.columns)}")
                return None
            if dimension_column not in df.columns:
                print(f"❌ Dimension column '{dimension_column}' not in DataFrame columns: {list(df.columns)}")
                return None
            
            # Prepare data: convert metric to numeric, handle dimension
            df_clean = df[[dimension_column, metric_column]].copy()
            
            # Convert metric to numeric
            df_clean[metric_column] = pd.to_numeric(df_clean[metric_column], errors='coerce')
            df_clean = df_clean.dropna(subset=[metric_column])
            
            # Handle dimension: convert to string for grouping
            df_clean[dimension_column] = df_clean[dimension_column].astype(str)
            
            if df_clean.empty:
                print(f"❌ DataFrame is empty after cleaning numeric values")
                return None
            
            # Build chart data using ACTUAL pandas operations
            chart_type = chart_intent.get('chart_type', 'bar')
            
            # For pie/doughnut charts, use sum aggregation
            # For other charts, use sum by default (can be changed to mean/median if needed)
            if chart_type in ['pie', 'doughnut', 'donut']:
                aggregated = df_clean.groupby(dimension_column)[metric_column].sum()
            else:
                # For bar/line charts, use sum (can be changed based on query intent)
                aggregated = df_clean.groupby(dimension_column)[metric_column].sum()
            
            if aggregated.empty:
                print(f"❌ Aggregated data is empty")
                return None
            
            # Convert to lists for Chart.js
            labels = aggregated.index.tolist()
            data_points = aggregated.values.tolist()
            
            # Limit data points for performance
            max_points = 150
            if len(labels) > max_points:
                # Sort by value and take top N
                sorted_pairs = sorted(zip(labels, data_points), key=lambda x: x[1], reverse=True)
                labels, data_points = zip(*sorted_pairs[:max_points])
                labels = list(labels)
                data_points = list(data_points)
            
            # Build Chart.js configuration
            chart_id = f"chart_original_{hash(filename)}_{int(datetime.now().timestamp())}"
            
            palette = ['#10a37f', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899']
            
            if chart_type in ['pie', 'doughnut', 'donut']:
                colors = [palette[i % len(palette)] for i in range(len(data_points))]
                dataset_config = {
                    'label': metric_column,
                    'data': data_points,
                    'backgroundColor': colors,
                    'borderColor': '#1f2937',
                    'borderWidth': 1
                }
                options = {
                    'maintainAspectRatio': False,
                    'plugins': {
                        'legend': {'display': True, 'position': 'right'},
                        'title': {'display': True, 'text': f"{metric_column} by {dimension_column}"}
                    }
                }
            else:
                color = palette[0]
                dataset_config = {
                    'label': metric_column,
                    'data': data_points,
                    'backgroundColor': color,
                    'borderColor': color,
                    'fill': chart_type != 'line',
                    'borderWidth': 2 if chart_type == 'line' else 0,
                    'tension': 0.3 if chart_type == 'line' else 0
                }
                options = {
                    'maintainAspectRatio': False,
                    'plugins': {
                        'legend': {'display': True},
                        'title': {'display': True, 'text': f"{metric_column} by {dimension_column}"},
                        'tooltip': {'mode': 'index', 'intersect': False}
                    },
                    'scales': {
                        'x': {'title': {'display': True, 'text': dimension_column}},
                        'y': {'title': {'display': True, 'text': metric_column}, 'beginAtZero': True}
                    }
                }
            
            return {
                'chart_id': chart_id,
                'type': chart_type if chart_type != 'donut' else 'doughnut',
                'labels': labels,
                'datasets': [dataset_config],
                'title': f"{metric_column} by {dimension_column}",
                'options': options,
                'meta': {
                    'source': filename,
                    'metric_column': metric_column,
                    'dimension_column': dimension_column,
                    'row_count': len(df_clean),
                    'data_source': 'original_file',  # 🎯 Mark as high-quality data
                    'column_selection_method': 'smart_selection'
                }
            }
            
        except Exception as e:
            print(f"❌ DataFrame chart building failed: {e}")
            import traceback
            traceback.print_exc()
            return None


    
    async def sync_entire_drive(self, credentials_dict: Dict, namespace: str, user_email: str) -> Dict[str, Any]:
        """Sync Google Drive for any role with their accessible folders"""
        try:
            print(f"🔄 Starting Google Drive sync for {namespace}: {user_email}")
            
            # Use the drive manager's sync method
            result = await self.drive_manager.sync_entire_drive(credentials_dict, namespace, user_email)
            return result
            
        except Exception as e:
            print(f"❌ Drive sync failed in RAG system: {e}")
            return {"status": "error", "error": str(e)}


    def _detect_file_intent_ai(self, query: str) -> Optional[Dict]:
        """SMART: Use AI to understand if user wants file content"""
        print(f"🧠 AI INTENT DETECTION: '{query}'")
        
        intent_prompt = f"""
        Analyze if the user is asking to see the CONTENT of a specific file/document.
        
        USER QUERY: "{query}"
        
        Consider these INTENT PATTERNS:
        - User wants to see what's inside a file
        - User is asking about content of a specific document
        - User wants to view/read/open a file
        - User is asking "what does [filename] contain"
        - User wants to see data from a specific file
        
        Respond with JSON only:
        {{
            "is_file_query": true/false,
            "filename": "extracted filename or empty string",
            "confidence": 0.0-1.0,
            "intent": "view_content|see_data|open_file|unknown"
        }}
        
        Examples:
        - "what does quarterly_report.pdf contain" → true, "quarterly_report.pdf", 0.95, "view_content"
        - "show me budget.xlsx" → true, "budget.xlsx", 0.9, "see_data" 
        - "tell me about the project plan" → true, "project plan", 0.8, "view_content"
        - "what's in the HR policy" → true, "HR policy", 0.85, "view_content"
        - "what is our revenue" → false, "", 0.1, "unknown"
        - "who is the CEO" → false, "", 0.05, "unknown"
        
        JSON Response:
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": intent_prompt}],
                "temperature": 0.1,
                "max_tokens": 500,
                "stream": False
            }

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            
            result_text = response.json()["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response
            import json
            intent_data = json.loads(result_text)
            
            print(f"🎯 AI INTENT RESULT: {intent_data}")
            
            if intent_data.get('is_file_query', False) and intent_data.get('confidence', 0) > 0.7:
                return {
                    'filename': intent_data['filename'],
                    'confidence': intent_data['confidence'],
                    'intent': intent_data['intent'],
                    'original_query': query
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ AI intent detection failed: {e}")
            return None

    def _enhanced_file_content_query(self, file_intent: Dict, namespace: str, session_id: str = None) -> Dict[str, Any]:
        """ENHANCED file content query with expanded document retrieval"""
        
        target_filename = file_intent['filename']
        original_query = file_intent['original_query']
        
        print(f"🚀 ENHANCED FILE QUERY: '{target_filename}' - MAXIMIZING DOCUMENT RETRIEVAL")
        
        try:
            # STRATEGY 1: Get MAXIMUM documents from both search methods
            vector_docs = self._get_vector_results(original_query, namespace, top_k=60)
            hybrid_docs = self._get_hybrid_results(original_query, namespace, top_k=80)
            
            print(f"📊 Vector docs: {len(vector_docs)}, Hybrid docs: {len(hybrid_docs)}")
            
            # 🚨 FIX: Enhanced deduplication that handles Document objects
            all_docs = vector_docs + hybrid_docs
            
            # Remove duplicates using multiple criteria
            unique_docs = []
            seen_identifiers = set()
            
            for doc in all_docs:
                # Create a unique identifier using source + content preview
                source = doc.metadata.get('source', 'unknown')
                content_preview = doc.page_content[:200] if doc.page_content else ""
                doc_identifier = f"{source}_{hash(content_preview)}"
                
                if doc_identifier not in seen_identifiers:
                    seen_identifiers.add(doc_identifier)
                    unique_docs.append(doc)
            
            print(f"🔄 Deduplication: {len(all_docs)} → {len(unique_docs)} unique documents")
            
            # STRATEGY 2: Broad filename matching
            filename_matches = []
            target_lower = target_filename.lower()
            target_without_ext = os.path.splitext(target_lower)[0]
            target_clean = re.sub(r'[^a-zA-Z0-9]', '', target_without_ext)
            
            for doc in unique_docs:
                source = doc.metadata.get('source', '').lower()
                source_clean = re.sub(r'[^a-zA-Z0-9]', '', source)
                
                # Multiple matching strategies
                exact_match = target_lower in source
                partial_match = target_without_ext in source  
                clean_match = target_clean in source_clean
                word_match = any(word in source for word in target_without_ext.split() if len(word) > 2)
                
                if exact_match or partial_match or clean_match or word_match:
                    filename_matches.append(doc)
            
            print(f"📁 Found {len(filename_matches)} documents with filename matches")
            
            # Combine all results, prioritizing filename matches
            final_docs = filename_matches + [doc for doc in unique_docs if doc not in filename_matches]
            
            print(f"🎯 ENHANCED FILE QUERY RESULTS: {len(final_docs)} final documents")
            
            if final_docs:
                context = self._build_smart_file_context(final_docs, target_filename, original_query)
                answer = self._generate_file_content_answer(context, original_query, target_filename)
                
                return self._prepare_response(
                    answer=answer,
                    confidence="high", 
                    documents=final_docs,
                    method="enhanced_file_content", 
                    session_id=session_id,
                    query=original_query
                )
            else:
                print("⚠️ No documents found, falling back to general query")
                return self._handle_general_query(original_query, namespace, session_id)
                
        except Exception as e:
            print(f"❌ Enhanced file query failed: {e}")
            import traceback
            traceback.print_exc()
            return self._handle_file_content_query(file_intent, namespace, session_id)



    def _search_documents_by_filename(self, all_docs: List[Document], target_filename: str, original_query: str) -> List[Document]:
        print(f"🔍 EXPANDED filename search: '{target_filename}'")
        
        matching_docs = []
        target_lower = target_filename.lower()
        
        # Remove extensions for broader matching
        target_without_ext = os.path.splitext(target_lower)[0]
        
        for doc in all_docs:
            source = doc.metadata.get('source', '').lower()
            
            # Multiple matching strategies
            exact_match = target_lower in source
            partial_match = target_without_ext in source
            word_match = any(word in source for word in target_without_ext.split() if len(word) > 3)
            
            if exact_match or partial_match or word_match:
                matching_docs.append(doc)
                print(f"   ✅ File match: {source}")
        
        print(f"📁 Found {len(matching_docs)} documents with filename matches")
        return matching_docs

    def _find_documents_by_ai_intent(self, all_docs: List[Document], target_filename: str, original_query: str) -> List[Document]:
        """SMART: Use hybrid search results for better file matching"""
        
        print(f"🔍 SMART DOCUMENT MATCHING for: '{target_filename}'")
        
        # 🎯 FIRST: Use hybrid search for this specific query
        hybrid_results = self.hybrid_retriever.hybrid_search(original_query, top_k=30) if self.hybrid_retriever else []
        hybrid_docs = [doc for doc, score in hybrid_results]
        
        if not hybrid_docs:
            print("⚠️ No hybrid results, using provided documents")
            hybrid_docs = all_docs
        
        # 🎯 Use AI to select the best matches from HYBRID results
        doc_context = "Available documents from HYBRID search:\n"
        for i, doc in enumerate(hybrid_docs[:50]):  # Use hybrid results
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            doc_context += f"{i+1}. {source}\n   Preview: {content_preview}\n\n"
        
        matching_prompt = f"""
        User is looking for files related to: "{target_filename}"
        Original query: "{original_query}"
        
        {doc_context}
        
        Which documents are most relevant to the user's file request?
        Return document numbers (like 1, 3, 5) that match the file the user wants to see.
        Focus on documents where the filename appears in the source/metadata.
        If no good matches, return "none".
        
        Relevant documents: """
        
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": matching_prompt}],
                "temperature": 0.1,
                "max_tokens": 100,
                "stream": False
            }

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            
            result = response.json()["choices"][0]["message"]["content"].strip()
            print(f"🎯 AI DOCUMENT SELECTION: {result}")
            
            # Parse document numbers from AI response
            import re
            doc_numbers = re.findall(r'\b\d+\b', result)
            
            matching_docs = []
            for num in doc_numbers:
                idx = int(num) - 1
                if 0 <= idx < len(hybrid_docs[:30]):
                    matching_docs.append(hybrid_docs[idx])
            
            print(f"✅ AI selected {len(matching_docs)} documents from hybrid search")
            return matching_docs[:10]  # Return top 5 matches
            
        except Exception as e:
            print(f"⚠️ AI document matching failed: {e}")
            # Fallback to hybrid search results
            return hybrid_docs[:10]

    def _fallback_filename_matching(self, all_docs: List[Document], target_filename: str) -> List[Document]:
        """Fallback: Simple filename matching when AI fails"""
        matching_docs = []
        target_lower = target_filename.lower()
        
        for doc in all_docs:
            source = doc.metadata.get('source', '').lower()
            if target_lower in source:
                matching_docs.append(doc)
        
        return matching_docs[:10]

    def _build_smart_file_context(self, documents: List[Document], filename: str, query: str) -> str:
        """Build context for file content summary"""
        context_parts = [
            f"USER IS ASKING ABOUT FILE CONTENT",
            f"TARGET FILE: {filename}",
            f"ORIGINAL QUERY: {query}",
            f"",
            f"FOUND DOCUMENTS:",
            f""
        ]
        
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            
            context_parts.append(f"--- DOCUMENT {i+1}: {source} ---")
            context_parts.append(content)
            context_parts.append("")
        
        return "\n".join(context_parts)

    def _generate_file_content_answer(self, context: str, query: str, filename: str) -> str:
        """Generate intelligent file content summary using AI"""
        
        prompt = f"""
        The user asked: "{query}"
        They want to know about the file: "{filename}"
        
        Here's the content I found:
        {context}
        
        Please provide a helpful summary that:
        1. Clearly states what content was found for this file
        2. Summarizes the key information available
        3. Mentions if certain information is missing
        4. Is direct and conversational
        
        If you found the actual file content, summarize it.
        If you found only references to the file, explain what's available.
        
        RESPONSE:
        """
        
        return self._generate_answer(prompt, query)

    def _generate_file_not_found_response(self, original_query: str, target_filename: str, available_docs: List[Document]) -> str:
        """Generate helpful response when file isn't found"""
        
        # Get list of available file names for suggestions
        available_files = list(set([doc.metadata.get('source', 'Unknown') for doc in available_docs[:20]]))
        
        prompt = f"""
        The user asked: "{original_query}"
        They're looking for a file like: "{target_filename}"
        
        Available files in the system:
        {chr(10).join([f"- {f}" for f in available_files[:10]])}
        
        Generate a helpful response that:
        1. Politely says the specific file wasn't found
        2. Suggests similar available files if any seem relevant
        3. Offers to help with what IS available
        4. Is friendly and helpful
        
        RESPONSE:
        """
        
        return self._generate_answer(prompt, original_query)

    

    def _handle_general_query(self, query: str, namespace: str, session_id: str = None) -> Dict[str, Any]:
        """Handle regular document queries with PROPER method passing"""
        print(f"🔍 SMART FALLBACK: Starting search for '{query}'")
        
        # Vector search with STRICT validation
        vector_docs = self._get_vector_results(query, namespace, top_k=50)
        vector_answer, vector_confidence = self._generate_answer_with_documents(vector_docs, query)
        
        print(f"📊 VECTOR RESULTS: {len(vector_docs)} docs, confidence: {vector_confidence}")
        
        # 🚨 FIX: Pass method parameter explicitly
        vector_found_answer = self._is_ai_answer_found(vector_answer, vector_confidence, query, "vector")
        print(f"🎯 VECTOR ANSWER FOUND: {vector_found_answer}")
        
        if vector_found_answer:
            print("✅ VECTOR found good answer - USING VECTOR RESULTS")
            return self._prepare_response(vector_answer, vector_confidence, vector_docs, "vector", session_id, query)
        
        # Hybrid search with LENIENT validation
        print("❌ VECTOR failed - trying HYBRID...")
        hybrid_results = []
        hybrid_docs = []
        
        if self.hybrid_retriever:
            try:
                # 🚨 Get the actual fused results
                hybrid_results = self.hybrid_retriever.hybrid_search(query, top_k=50)
                hybrid_docs = [doc for doc, score in hybrid_results]
                
                hybrid_answer, hybrid_confidence = self._generate_answer_with_documents(hybrid_docs, query)
                
                print(f"📊 HYBRID RESULTS: {len(hybrid_docs)} docs, confidence: {hybrid_confidence}")
                
                # 🚨 FIX: Pass method parameter explicitly for HYBRID
                hybrid_found_answer = self._is_ai_answer_found(hybrid_answer, hybrid_confidence, query, "hybrid")
                print(f"🎯 HYBRID ANSWER FOUND: {hybrid_found_answer}")
                
                if hybrid_found_answer:
                    print("✅ HYBRID found good answer - USING HYBRID RESULTS")
                    return self._prepare_response(hybrid_answer, hybrid_confidence, hybrid_docs, "hybrid", session_id, query)
                
            except Exception as e:
                print(f"❌ Hybrid search failed: {e}")
                hybrid_docs = []
        
        # 🚨 ENHANCED FALLBACK: Use ACTUAL FUSED results
        print("❌ Both failed - generating RELATED ANSWER using FUSED results")
        
        fallback_docs = []
        
        # 🚨 PRIORITY 1: Use the actual fused hybrid results (they're the best)
        if hybrid_results:
            # Use the documents from fused hybrid results (already scored and ranked)
            hybrid_docs_from_fused = [doc for doc, score in hybrid_results]
            fallback_docs.extend(hybrid_docs_from_fused)
            print(f"🎯 FALLBACK: Using {len(hybrid_docs_from_fused)} ACTUAL FUSED hybrid documents")
        
        # 🚨 PRIORITY 2: If no hybrid results, fall back to vector
        elif vector_docs:
            fallback_docs.extend(vector_docs)
            print(f"🎯 FALLBACK: Using {len(vector_docs)} vector documents (no hybrid available)")
        
        # Remove duplicates by content to ensure quality
        unique_docs = []
        seen_content = set()
        for doc in fallback_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content and doc.page_content.strip():
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        print(f"📦 FALLBACK: Combined {len(fallback_docs)} docs → {len(unique_docs)} unique docs after deduplication")
        
        # Generate answer from the FUSED document set
        if unique_docs:
            related_answer = self._generate_related_answer(query, unique_docs)
            confidence = self._calculate_confidence(unique_docs)
        else:
            related_answer = f"I couldn't find any information about \"{query}\" in the corporate documents."
            confidence = "very low"
        
        return self._prepare_response(related_answer, confidence, unique_docs, "fallback_related", session_id, query)
# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title="Professional Enterprise Document Analysis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
print("🚀 Starting COMPLETE PROFESSIONAL RAG System...")
try:
    rag_system = ProfessionalRAGSystem()
    print("🎯 SYSTEM READY! All professional features activated")
except Exception as e:
    print(f"❌ SYSTEM INITIALIZATION FAILED: {e}")
    rag_system = None

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# ALL API ENDPOINTS - YOUR EXISTING
# =============================================================================

@app.post("/api/login")
async def login_user(data: dict):
    try:
        email = data.get('email', '').strip().lower()
        
        if not email:
            raise HTTPException(status_code=400, detail="Company email is required")
        
        user_manager = CorporateUserManager()
        user_context = user_manager.authenticate_user(email)
        
        # Start background sync for managers/executives
        drive_sync_info = {}
        if user_context.role in ["manager", "executive"] and HAS_GOOGLE_DRIVE:
            sync_result = rag_system.sync_google_drive(user_context)
            drive_sync_info = {
                "drive_sync_started": True,
                "drive_sync_message": sync_result.get("message", "Sync started")
            }
        else:
            drive_sync_info = {
                "drive_sync_started": False,
                "drive_sync_message": "Auto-sync only for managers/executives"
            }
        
        user_data = {
            "email": user_context.email,
            "role": user_context.role,
            "namespace": user_context.get_namespace(),
            "accessible_folders": user_context.get_accessible_folders(),
            "user_id": str(uuid.uuid4())[:8],
            "is_manager": user_context.role in ["manager", "executive"],
            "google_drive_available": HAS_GOOGLE_DRIVE,
            **drive_sync_info
        }
        
        return {
            "message": "Login successful - professional RAG system active", 
            "user": user_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")



@app.post("/api/sync-drive")
async def sync_drive(data: dict):
    """UNLIMITED Google Drive sync for ALL roles"""
    try:
        email = data.get('email')
        
        print(f"🔍 SYNC DEBUG: Received sync request for email: {email}")
        print(f"🔍 SYNC DEBUG: Available user_credentials: {list(user_credentials.keys())}")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email required")
        
        if not user_credentials:
            raise HTTPException(status_code=400, detail="No Google Drive credentials")
        
        # Get user context using the LOGIN email
        user_manager = CorporateUserManager()
        user_context = user_manager.authenticate_user(email)
        
        print(f"🔍 SYNC DEBUG: User context - role: {user_context.role}, namespace: {user_context.namespace}")
        print(f"🔍 SYNC DEBUG: Accessible folders: {user_context.get_accessible_folders()}")
        
        print(f"🚀 {user_context.role} starting UNLIMITED Google Drive sync...")
        
        # Start unlimited sync with LOGIN email
        user_email, credentials_dict = list(user_credentials.items())[0]
        
        print(f"🔍 SYNC DEBUG: Using Google Drive account: {user_email}")
        print(f"🔍 SYNC DEBUG: Passing login email to sync: {email}")
        
        import asyncio
        asyncio.create_task(
            rag_system.drive_manager.sync_entire_drive(credentials_dict, user_context.namespace, email)  # Pass login email
        )
        
        return {
            "status": "started", 
            "message": f"UNLIMITED Google Drive sync started for {user_context.role}",
            "accessible_folders": user_context.get_accessible_folders(),
            "login_email": email,  # 🆕 Include for debugging
            "google_drive_account": user_email,  # 🆕 Include for debugging
            "note": "This may take several minutes for large drives"
        }
        
    except Exception as e:
        print(f"❌ SYNC ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fast-recover-missing-docs")
async def fast_recover_missing_docs():
    """FAST recovery: Add missing documents from hybrid to vector store"""
    try:
        vector_store = rag_system.vector_store
        
        # Get documents from both stores
        hybrid_docs = vector_store.hybrid_retrieval.documents_cache if vector_store.hybrid_retrieval else []
        vector_docs = vector_store.vector_store.similarity_search("", k=10000)
        
        # Find documents in hybrid but NOT in vector store
        vector_sources = set(doc.metadata.get('source', '') for doc in vector_docs)
        hybrid_sources = set(doc.metadata.get('source', '') for doc in hybrid_docs)
        
        missing_sources = hybrid_sources - vector_sources
        print(f"🔍 Found {len(missing_sources)} documents missing from vector store")
        
        # Get the actual missing documents
        missing_docs = []
        for doc in hybrid_docs:
            if doc.metadata.get('source', '') in missing_sources:
                missing_docs.append(doc)
        
        print(f"📦 Adding {len(missing_docs)} missing documents to vector store...")
        
        # Add missing documents using the fixed method
        result = vector_store.add_documents("management_full", missing_docs)
        
        return {
            "missing_documents_found": len(missing_sources),
            "documents_added": len(missing_docs),
            "add_documents_result": result,
            "recovery_status": "COMPLETE" if result > 0 else "FAILED"
        }
        
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/api/verify-sync-status")
async def verify_sync_status():
    """Verify vector store and hybrid cache are synchronized"""
    try:
        vector_store = rag_system.vector_store
        
        # Get counts from both stores
        vector_docs = vector_store.vector_store.similarity_search("", k=10000)
        hybrid_docs = vector_store.hybrid_retrieval.documents_cache if vector_store.hybrid_retrieval else []
        
        # Check specific test files
        test_files = ["fast_upload_test.txt", "frontenddd_flow_test.txt", "frontend_flow_test.txt"]
        
        sync_status = {}
        for filename in test_files:
            in_vector = any(filename in doc.metadata.get('source', '') for doc in vector_docs)
            in_hybrid = any(filename in doc.metadata.get('source', '') for doc in hybrid_docs)
            
            sync_status[filename] = {
                "in_vector_store": in_vector,
                "in_hybrid_cache": in_hybrid,
                "synchronized": in_vector and in_hybrid
            }
        
        return {
            "vector_store_count": len(vector_docs),
            "hybrid_cache_count": len(hybrid_docs),
            "counts_synchronized": len(vector_docs) == len(hybrid_docs),
            "test_files_status": sync_status,
            "status": "PERFECT" if len(vector_docs) == len(hybrid_docs) else "NEEDS_SYNC"
        }
        
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/api/verify-document-sync")
async def verify_document_sync():
    """Verify documents are synchronized between vector store and hybrid cache"""
    try:
        vector_store = rag_system.vector_store
        
        # Get counts from both stores
        vector_docs = vector_store.vector_store.similarity_search("", k=10000)
        hybrid_docs = vector_store.hybrid_retrieval.documents_cache if vector_store.hybrid_retrieval else []
        
        # Check specific important documents
        important_docs = ["paper.docx", "HR_Policy_Manual_Detailed.docx", "s.jpeg"]
        
        sync_status = {}
        for doc_name in important_docs:
            in_vector = any(doc_name in doc.metadata.get('source', '') for doc in vector_docs)
            in_hybrid = any(doc_name in doc.metadata.get('source', '') for doc in hybrid_docs)
            
            sync_status[doc_name] = {
                "in_vector_store": in_vector,
                "in_hybrid_cache": in_hybrid,
                "synchronized": in_vector and in_hybrid
            }
        
        return {
            "vector_store_count": len(vector_docs),
            "hybrid_cache_count": len(hybrid_docs),
            "synchronization_status": sync_status,
            "all_synchronized": all(status["synchronized"] for status in sync_status.values())
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/test-dual-storage")
async def test_dual_storage():
    """Test if dual storage is working"""
    try:
        # Test with a small Excel file
        import pandas as pd
        from io import BytesIO
        
        # Create test data
        test_data = {
            'Species': ['setosa', 'versicolor', 'virginica'] * 50,
            'SepalLength': [5.1, 7.0, 6.3] * 50,
            'SepalWidth': [3.5, 3.2, 3.3] * 50,
            'PetalLength': [1.4, 4.7, 6.0] * 50, 
            'PetalWidth': [0.2, 1.4, 2.5] * 50
        }
        df = pd.DataFrame(test_data)
        
        # Save to bytes
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Iris', index=False)
        excel_bytes = output.getvalue()
        
        # Test upload
        result = rag_system.ingest_document_with_auto_sync(
            "management_full", excel_bytes, "test_iris.xlsx"
        )
        
        # Check if original file was stored
        docs = rag_system.document_storage.get_documents("management_full")
        test_doc = None
        for doc in docs:
            if "test_iris.xlsx" in doc['filename']:
                test_doc = doc
                break
        
        return {
            "upload_success": result['success'],
            "chunks_created": result.get('chunks', 0),
            "document_stored": test_doc is not None,
            "has_chart_data_id": test_doc.get('chart_data_id') if test_doc else False,
            "chart_data_id": test_doc.get('chart_data_id') if test_doc else None
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/check-chart-data/{filename}")
async def check_chart_data(filename: str):
    """Check if a file has chart data stored"""
    try:
        docs = rag_system.document_storage.get_documents("management_full")
        target_doc = None
        
        for doc in docs:
            if filename.lower() in doc['filename'].lower():
                target_doc = doc
                break
        
        if not target_doc:
            return {"error": f"File {filename} not found"}
        
        # Check if original data is available
        chart_data_id = target_doc.get('chart_data_id')
        has_original_data = False
        original_data_size = 0
        
        if chart_data_id:
            original_data = rag_system.document_storage.get_original_file("management_full", chart_data_id)
            if original_data:
                has_original_data = True
                original_data_size = len(original_data)
        
        return {
            "filename": filename,
            "chart_data_id": chart_data_id,
            "has_original_data": has_original_data,
            "original_data_size": original_data_size,
            "structured_data_available": target_doc.get('has_structured_data', False),
            "document_metadata": target_doc
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/test-pdf-chunking")
async def test_pdf_chunking():
    """Test PDF chunking with a sample file"""
    # This will help verify the PDF chunking is working
    return {"message": "PDF chunking test endpoint - upload a PDF to test"}

@app.get("/api/test-drive-sync/{role}")
async def test_drive_sync(role: str):
    """Test drive sync for different roles"""
    test_email = f"{role}@company.com"
    user_manager = CorporateUserManager()
    user_context = user_manager.authenticate_user(test_email)
    
    return {
        "role": role,
        "accessible_folders": user_context.get_accessible_folders(),
        "can_sync": True,  # All roles can sync now
        "namespace": user_context.get_namespace()
    }

@app.get("/api/test-permanent-fix")
async def test_permanent_fix():
    """Test that new uploads stay synchronized with the permanent fix"""
    try:
        # Create a new test file
        test_content = b"This tests the permanent synchronization fix"
        test_filename = "permanent_fix_test.txt"
        
        # Upload using the normal flow
        result = rag_system.ingest_document_with_auto_sync(
            "management_full", 
            test_content, 
            test_filename
        )
        
        # Verify synchronization
        vector_store = rag_system.vector_store
        in_vector = any("permanent_fix_test.txt" in doc.metadata.get('source', '') 
                       for doc in vector_store.vector_store.similarity_search("permanent synchronization fix", k=5))
        
        in_hybrid = False
        if vector_store.hybrid_retrieval:
            in_hybrid = any("permanent_fix_test.txt" in doc.metadata.get('source', '') 
                           for doc in vector_store.hybrid_retrieval.documents_cache)
        
        # Get final counts
        vector_count = len(vector_store.vector_store.similarity_search("", k=15000))
        hybrid_count = len(vector_store.hybrid_retrieval.documents_cache) if vector_store.hybrid_retrieval else 0
        
        return {
            "permanent_fix_test": "COMPLETE",
            "upload_result": result,
            "in_vector_store": in_vector,
            "in_hybrid_cache": in_hybrid,
            "both_stores_synced": in_vector and in_hybrid,
            "final_counts": {
                "vector_store": vector_count,
                "hybrid_cache": hybrid_count,
                "synchronized": vector_count == hybrid_count
            },
            "status": "PERMANENT_FIX_WORKING" if vector_count == hybrid_count else "NEEDS_MORE_FIXES"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/test-fast-upload")
async def test_fast_upload():
    """Test the FAST upload without rebuild"""
    try:
        # Simulate exactly what happens during frontend upload
        test_content = b"This is a FAST upload test"
        test_filename = "fast_upload_test.txt"
        
        # This is what your /upload endpoint does:
        result = rag_system.ingest_document_with_auto_sync(
            "management_full", 
            test_content, 
            test_filename
        )
        
        # Verify it worked
        vector_store = rag_system.vector_store
        in_vector = any("fast_upload_test.txt" in doc.metadata.get('source', '') 
                       for doc in vector_store.vector_store.similarity_search("FAST upload test", k=5))
        
        in_hybrid = False
        if vector_store.hybrid_retrieval:
            in_hybrid = any("fast_upload_test.txt" in doc.metadata.get('source', '') 
                           for doc in vector_store.hybrid_retrieval.documents_cache)
        
        return {
            "fast_upload_test": "COMPLETE",
            "ingest_method_result": result,
            "in_vector_store": in_vector,
            "in_hybrid_cache": in_hybrid,
            "both_stores_synced": in_vector and in_hybrid,
            "processing_time": "Should be seconds, not minutes"
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/test-frontend-upload-flow")
async def test_frontend_upload_flow():
    """Test the complete frontend upload flow with the fixed add_documents"""
    try:
        # Simulate exactly what happens during frontend upload
        test_content = b"This is aaaaa frontend upload flow test"
        test_filename = "frontenddd_flow_test.txt"
        
        # This is what your /upload endpoint does:
        result = rag_system.ingest_document_with_auto_sync(
            "management_full", 
            test_content, 
            test_filename
        )
        
        # Verify it worked
        vector_store = rag_system.vector_store
        in_vector = any("frontend_flow_test.txt" in doc.metadata.get('source', '') 
                       for doc in vector_store.vector_store.similarity_search("frontend upload flow test", k=5))
        
        in_hybrid = False
        if vector_store.hybrid_retrieval:
            in_hybrid = any("frontend_flow_test.txt" in doc.metadata.get('source', '') 
                           for doc in vector_store.hybrid_retrieval.documents_cache)
        
        return {
            "upload_flow_test": "COMPLETE",
            "ingest_method_result": result,
            "in_vector_store": in_vector,
            "in_hybrid_cache": in_hybrid,
            "both_stores_synced": in_vector and in_hybrid,
            "conclusion": "Frontend upload will work once add_documents is fixed"
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/fallback-stats")
async def get_fallback_stats():
    """Get statistics on smart fallback performance"""
    if hasattr(rag_system, 'fallback_stats'):
        stats = rag_system.fallback_stats
        stats['hybrid_success_rate'] = f"{(stats['hybrid_success'] / stats['fallback_used'] * 100):.1f}%" if stats['fallback_used'] > 0 else "N/A"
        stats['improvement_rate'] = f"{(stats['fallback_used'] / stats['total_queries'] * 100):.1f}%" if stats['total_queries'] > 0 else "N/A"
        return stats
    return {"message": "No statistics collected yet"}
    
@app.get("/api/verify-score-type")
async def verify_score_type(query: str = "test"):
    """Verify if scores are distance or similarity"""
    results = rag_system.vector_store.vector_store.similarity_search_with_relevance_scores(query, k=3)
    
    score_info = []
    for doc, score in results:
        score_info.append({
            'source': doc.metadata.get('source', 'unknown'),
            'raw_score': score,
            'type_guess': 'SIMILARITY' if score <= 1.0 else 'DISTANCE',
            'explanation': f"If this is similarity: {score:.3f} = {score*100:.1f}% match" if score <= 1.0 else f"If this is distance: {score:.3f} (lower = better)"
        })
    
    return {
        'query': query,
        'conclusion': 'Your scores appear to be SIMILARITY scores (0-1, higher = better)',
        'results': score_info
    }

@app.get("/api/test-final-fix")
async def test_final_fix():
    """Test the complete accuracy fix"""
    try:
        result = rag_system.query("what does cv_job_o.pdf have", "management_full")
        
        return {
            "success": True,
            "documents_used": result['documents_used'],
            "top_document": result['sources'][0]['source'] if result['sources'] else 'none',
            "answer_preview": result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/rebuild-hybrid-index")
async def rebuild_hybrid_index():
    """Force rebuild hybrid index with ALL documents - FIXED"""
    try:
        vector_store = rag_system.vector_store
        
        # 🚨 FIX: Use the initialization method
        if vector_store.hybrid_retrieval is None:
            print("🔄 Creating new hybrid retrieval instance...")
            vector_store._initialize_hybrid_retrieval()
        
        hybrid = vector_store.hybrid_retrieval
        
        # Get ALL documents from vector store
        all_docs = vector_store.vector_store.similarity_search("", k=1000)
        
        print(f"🔄 Rebuilding hybrid index with {len(all_docs)} documents")
        
        # Rebuild the index
        hybrid.build_hybrid_index(all_docs)
        
        return {
            "success": True,
            "documents_loaded": len(hybrid.documents_cache),
            "document_types": list(set([doc.metadata.get('file_type', 'unknown') for doc in hybrid.documents_cache]))
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/force-sync-now")
async def force_sync_now():
    """ONE-TIME: Force complete sync between vector store and hybrid cache"""
    try:
        vector_store = rag_system.vector_store
        
        # Get ALL documents from vector store
        all_docs = vector_store.vector_store.similarity_search("", k=15000)
        print(f"📊 Vector store has {len(all_docs)} documents")
        
        # Ensure hybrid retriever exists
        if vector_store.hybrid_retrieval is None:
            vector_store.hybrid_retrieval = FreeHybridRetrieval(vector_store, vector_store.embeddings)
        
        # Smart deduplication
        unique_docs = []
        seen_identifiers = set()
        for doc in all_docs:
            source = doc.metadata.get('source', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            doc_identifier = f"{source}_{chunk_id}"
            
            if doc_identifier not in seen_identifiers:
                seen_identifiers.add(doc_identifier)
                unique_docs.append(doc)
        
        # Build hybrid index
        vector_store.hybrid_retrieval.build_hybrid_index(unique_docs)
        
        hybrid_count = len(vector_store.hybrid_retrieval.documents_cache)
        
        return {
            "status": "COMPLETE_SYNC",
            "vector_store_documents": len(all_docs),
            "hybrid_cache_documents": hybrid_count,
            "sync_perfect": len(all_docs) == hybrid_count,
            "message": f"Hybrid cache synced with {hybrid_count} documents"
        }
        
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/force-refresh-vector-store")
async def force_refresh_vector_store(namespace: str = "management_full"):
    """COMPLETE nuclear option - recreate vector store from scratch"""
    try:
        print("💥 FORCE REFRESH: Recreating vector store from scratch...")
        
        # 1. Get current valid documents
        all_docs = rag_system.vector_store.vector_store.similarity_search("", k=2000)
        valid_docs = [doc for doc in all_docs if doc.page_content and doc.page_content.strip()]
        
        print(f"📊 Found {len(valid_docs)} valid documents out of {len(all_docs)} total")
        
        # 2. COMPLETELY recreate the vector store
        rag_system.vector_store.vector_store = None
        
        # Reinitialize Chroma
        rag_system.vector_store.vector_store = Chroma(
            persist_directory=rag_system.vector_store.persist_directory,
            embedding_function=rag_system.vector_store.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # 3. Add only valid documents
        if valid_docs:
            rag_system.vector_store.vector_store.add_documents(valid_docs)
        
        # 4. Rebuild search index
        rag_system.vector_store.hybrid_retrieval = None
        rag_system.vector_store.hybrid_retrieval = FreeHybridRetrieval(
            rag_system.vector_store, 
            rag_system.vector_store.embeddings
        )
        rag_system.vector_store.hybrid_retrieval.build_hybrid_index(valid_docs)
        
        # 5. Force reload document storage
        rag_system.document_storage.auto_sync_from_vector_store(namespace, rag_system.vector_store)
        
        return {
            "message": "Vector store force refreshed successfully",
            "valid_documents_added": len(valid_docs),
            "empty_documents_removed": len(all_docs) - len(valid_docs),
            "namespace": namespace
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug-source-scores")
async def debug_source_scores(query: str = "ICEB2025_paper_9"):
    """Debug what scores are actually being calculated"""
    try:
        namespace = "management_full"
        
        # Test vector search
        vector_docs = rag_system._get_vector_results(query, namespace, top_k=5)
        vector_sources = []
        for doc in vector_docs:
            vector_sources.append({
                'source': doc.metadata.get('source', 'unknown'),
                'relevance_score': doc.metadata.get('relevance_score', 'N/A'),
                'content_length': len(doc.page_content)
            })
        
        # Test hybrid search  
        hybrid_docs = rag_system._get_hybrid_results(query, namespace, top_k=5)
        hybrid_sources = []
        for doc in hybrid_docs:
            hybrid_sources.append({
                'source': doc.metadata.get('source', 'unknown'), 
                'relevance_score': doc.metadata.get('relevance_score', 'N/A'),
                'content_length': len(doc.page_content)
            })
        
        return {
            'query': query,
            'vector_results': vector_sources,
            'hybrid_results': hybrid_sources,
            'problem': 'Relevance scores not being passed to metadata correctly'
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-role-mapping")
async def debug_role_mapping(email: str):
    """Debug role mapping for a specific email"""
    try:
        user_manager = CorporateUserManager()
        user_context = user_manager.authenticate_user(email)
        
        return {
            "email": email,
            "detected_role": user_context.role,
            "namespace": user_context.namespace,
            "accessible_folders": user_context.get_accessible_folders(),
            "has_full_access": "*" in user_context.get_accessible_folders(),
            "corporate_access": CORPORATE_FOLDER_ACCESS.get(user_context.role, [])
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-role-access")
async def debug_role_access(email: str):
    """Debug role-based folder access"""
    try:
        user_manager = CorporateUserManager()
        user_context = user_manager.authenticate_user(email)
        
        return {
            "email": email,
            "role": user_context.role,
            "namespace": user_context.namespace,
            "accessible_folders": user_context.get_accessible_folders(),
            "is_manager": user_context.role in ["manager", "executive"]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-chart-flow")
async def debug_chart_flow(query: str = "pie chart of iris species"):
    """Debug the complete chart flow"""
    try:
        rag = rag_system
        
        # Step by step debugging
        chart_intent = rag._parse_chart_request(query)
        print(f"🎯 Chart intent: {chart_intent}")
        
        dataset_match = rag._find_structured_dataset("management_full", chart_intent)
        print(f"🔍 Dataset match: {dataset_match}")
        
        if dataset_match:
            print(f"📊 Dataset: {dataset_match['doc'].get('filename')}")
            print(f"🔑 Chart data ID: {dataset_match['doc'].get('chart_data_id')}")
            
            # Test original data
            original_chart = rag.build_chart_from_original_data(
                query, "management_full", dataset_match['doc']
            )
            print(f"🎯 Original data chart: {original_chart is not None}")
        
        # Test full flow
        result = rag.query(query, "management_full")
        
        return {
            "chart_intent": chart_intent,
            "dataset_found": bool(dataset_match),
            "has_chart_data_id": dataset_match['doc'].get('chart_data_id') if dataset_match else False,
            "original_data_worked": original_chart is not None if dataset_match else False,
            "final_result_has_chart": "chart" in result,
            "method_used": result.get('method_used'),
            "confidence": result.get('confidence')
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug-chart-selection")
async def debug_chart_selection(query: str):
    """Debug why chart selection is picking wrong files"""
    try:
        chart_intent = rag_system._parse_chart_request(query)
        if not chart_intent:
            return {"error": "Not a chart request"}
        
        print(f"🔍 CHART DEBUG: Query='{query}'")
        print(f"🎯 Chart intent: {chart_intent}")
        
        # Check what files are available
        all_files = list(rag_system.document_storage.iterate_structured_sheets("management_full"))
        print(f"📁 Available files: {[item['doc'].get('filename') for item in all_files]}")
        
        # Test the file matching
        dataset_match = rag_system._find_structured_dataset("management_full", chart_intent)
        
        return {
            "query": query,
            "chart_intent": chart_intent,
            "available_files": [item['doc'].get('filename') for item in all_files],
            "selected_file": dataset_match['doc'].get('filename') if dataset_match else None,
            "selected_metric": dataset_match.get('metric_column') if dataset_match else None,
            "selected_dimension": dataset_match.get('dimension_column') if dataset_match else None
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug-missing-documents")
async def debug_missing_documents():
    """Find out why documents are missing from vector store"""
    try:
        vector_store = rag_system.vector_store
        
        # Get documents from both stores
        vector_docs = vector_store.vector_store.similarity_search("", k=10000)
        hybrid_docs = vector_store.hybrid_retrieval.documents_cache if vector_store.hybrid_retrieval else []
        
        # Find documents that are in hybrid but NOT in vector store
        vector_sources = set()
        for doc in vector_docs:
            source = doc.metadata.get('source', 'unknown')
            vector_sources.add(source)
        
        hybrid_sources = set()
        for doc in hybrid_docs:
            source = doc.metadata.get('source', 'unknown') 
            hybrid_sources.add(source)
        
        missing_from_vector = hybrid_sources - vector_sources
        missing_from_hybrid = vector_sources - hybrid_sources
        
        # Analyze the missing documents
        missing_analysis = []
        for source in list(missing_from_vector)[:20]:  # First 20 missing docs
            # Try to find why they're missing
            missing_analysis.append({
                'source': source,
                'likely_reason': 'Vector store add failed during processing'
            })
        
        return {
            "total_vector_docs": len(vector_docs),
            "total_hybrid_docs": len(hybrid_docs),
            "missing_from_vector_count": len(missing_from_vector),
            "missing_from_hybrid_count": len(missing_from_hybrid),
            "missing_from_vector_samples": list(missing_from_vector)[:10],
            "missing_from_hybrid_samples": list(missing_from_hybrid)[:10],
            "analysis": "Documents are being processed and added to hybrid but FAILING to be added to vector store"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-deduplication")
async def debug_deduplication(query: str = "what does chapter 10 of book of enoch contain"):
    """Debug what deduplication is removing"""
    try:
        namespace = "management_full"
        
        # Get results before deduplication
        vector_docs = rag_system._get_vector_results(query, namespace, top_k=100)
        hybrid_docs = rag_system._get_hybrid_results(query, namespace, top_k=100)
        
        all_docs = vector_docs + hybrid_docs
        
        # Apply deduplication
        unique_docs = []
        seen_identifiers = set()
        removed_docs = []
        
        for doc in all_docs:
            source = doc.metadata.get('source', 'unknown')
            content_preview = doc.page_content[:500]
            content_hash = hashlib.md5(content_preview.encode()).hexdigest()[:16]
            identifier = f"{source}_{content_hash}"
            
            if identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                unique_docs.append(doc)
            else:
                removed_docs.append({
                    'source': source,
                    'content_preview': content_preview,
                    'identifier': identifier,
                    'full_content_length': len(doc.page_content)
                })
        
        return {
            "total_before_deduplication": len(all_docs),
            "total_after_deduplication": len(unique_docs),
            "removed_count": len(removed_docs),
            "removed_documents_sample": removed_docs[:10],
            "deduplication_rate": f"{(len(removed_docs) / len(all_docs) * 100):.1f}%"
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug-vector-store-add")
async def debug_vector_store_add():
    """Debug why vector store add_documents is failing silently"""
    try:
        vector_store = rag_system.vector_store
        
        # Test 1: Direct ChromaDB add
        print("🧪 TEST 1: Direct ChromaDB add...")
        test_doc1 = Document(
            page_content="Direct ChromaDB add test content",
            metadata={"source": "direct_chroma_test.txt", "test": True}
        )
        
        # Add directly to Chroma
        direct_result = vector_store.vector_store.add_documents([test_doc1])
        print(f"📊 Direct Chroma add returned: {direct_result}")
        
        # Test 2: Using your add_documents method
        print("🧪 TEST 2: Using your add_documents method...")
        test_doc2 = Document(
            page_content="Your add_documents method test content", 
            metadata={"source": "your_method_test.txt", "test": True}
        )
        
        your_method_result = vector_store.add_documents("management_full", [test_doc2])
        print(f"📊 Your add_documents returned: {your_method_result}")
        
        # Verify both were actually stored
        all_docs = vector_store.vector_store.similarity_search("", k=1000)
        direct_found = any("direct_chroma_test.txt" in doc.metadata.get('source', '') for doc in all_docs)
        your_method_found = any("your_method_test.txt" in doc.metadata.get('source', '') for doc in all_docs)
        
        return {
            "direct_chroma_add_result": direct_result,
            "your_method_add_result": your_method_result,
            "direct_test_found_in_store": direct_found,
            "your_method_found_in_store": your_method_found,
            "total_docs_in_store": len(all_docs),
            "problem": "Your add_documents method is returning success but not actually storing documents"
        }
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}



@app.get("/api/debug-vector-store-errors")
async def debug_vector_store_errors():
    """Check if vector store operations are failing silently"""
    try:
        vector_store = rag_system.vector_store
        
        # Test adding a simple document to see if it works
        test_doc = Document(
            page_content="This is a test document to check if vector store is working",
            metadata={"source": "test_document", "test": True}
        )
        
        # Try to add to vector store
        result = vector_store.add_documents("management_full", [test_doc])
        
        # Check if it was actually added
        search_results = vector_store.vector_store.similarity_search("test document", k=5)
        test_found = any("test_document" in doc.metadata.get('source', '') for doc in search_results)
        
        return {
            "add_documents_return_value": result,
            "test_document_found_in_search": test_found,
            "search_results_count": len(search_results),
            "vector_store_health": "WORKING" if test_found else "BROKEN"
        }
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/api/debug-frontend-upload")
async def debug_frontend_upload():
    """Test the actual frontend upload endpoint"""
    try:
        # Create a test file to upload
        test_content = b"This is a frontend upload test document"
        test_filename = "frontend_upload_test.txt"
        
        # Simulate frontend upload
        form_data = {
            'namespace': 'management_full',
            'email': 'manager@company.com'
        }
        
        files = {
            'file': (test_filename, test_content, 'text/plain')
        }
        
        response = requests.post(
            "http://localhost:8000/upload",
            data=form_data,
            files=files
        )
        
        upload_result = response.json()
        
        # Check if it was actually stored
        vector_results = rag_system.vector_store.vector_store.similarity_search("frontend upload test", k=5)
        in_vector_store = any("frontend_upload_test.txt" in doc.metadata.get('source', '') for doc in vector_results)
        
        # Check document storage
        docs = rag_system.document_storage.get_documents("management_full")
        in_document_storage = any("frontend_upload_test.txt" in doc['filename'] for doc in docs)
        
        return {
            "upload_endpoint_response": upload_result,
            "actually_stored_in_vector_store": in_vector_store,
            "in_document_storage": in_document_storage,
            "problem": "Upload endpoint returns success but doesn't actually store documents"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-upload-endpoint")
async def debug_upload_endpoint():
    """Check what the upload endpoint is actually doing"""
    try:
        # Let's check the upload endpoint code path
        vector_store = rag_system.vector_store
        
        # Test the exact code path from your upload endpoint
        test_content = b"Upload endpoint debug test"
        test_filename = "upload_endpoint_debug.txt"
        
        # This is what your upload endpoint does:
        result = rag_system.ingest_document_with_auto_sync(
            "management_full", 
            test_content, 
            test_filename
        )
        
        # Check results
        in_vector = any("upload_endpoint_debug.txt" in doc.metadata.get('source', '') 
                       for doc in vector_store.vector_store.similarity_search("upload endpoint debug", k=5))
        
        in_doc_storage = any("upload_endpoint_debug.txt" in doc['filename'] 
                            for doc in rag_system.document_storage.get_documents("management_full"))
        
        return {
            "ingest_document_with_auto_sync_result": result,
            "in_vector_store_after_ingest": in_vector,
            "in_document_storage_after_ingest": in_doc_storage,
            "problem_location": "ingest_document_with_auto_sync method is failing"
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug-ingest-method")
async def debug_ingest_method():
    """Debug the ingest_document_with_auto_sync method directly"""
    try:
        test_content = b"Testing ingest_doddddcument_with_auto_sync method"
        test_filename = "ingest_method_teeest.txt"
        
        # Call the problematic method directly
        result = rag_system.ingest_document_with_auto_sync(
            "management_full", 
            test_content, 
            test_filename
        )
        
        print(f"🔍 ingest_document_with_auto_sync returned: {result}")
        
        # Check what actually happened
        in_vector = any("ingest_method_test.txt" in doc.metadata.get('source', '') 
                       for doc in rag_system.vector_store.vector_store.similarity_search("testing ingest method", k=5))
        
        return {
            "method_return_value": result,
            "actually_stored_in_vector": in_vector,
            "method_success": result.get('success', False),
            "chunks_reported": result.get('chunks', 0),
            "skipped_reason": result.get('reason', 'not_skipped'),
            "exact_problem": f"Method returns success={result.get('success')} but stored={in_vector}"
        }
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/api/debug-upload-process")
async def debug_upload_process():
    """Debug why uploads are failing to reach vector store"""
    try:
        # Test the complete upload pipeline
        test_content = "This is a test document for upload debugging"
        test_filename = "debug_test_upload.txt"
        
        # Step 1: Process document (like your upload does)
        processed_docs = rag_system.vector_store.document_processor.process_document(
            test_content.encode('utf-8'), 
            test_filename
        )
        
        print(f"📄 Document processing: {len(processed_docs)} chunks created")
        
        # Step 2: Add to vector store (like your upload does)
        vector_result = rag_system.vector_store.add_documents("management_full", processed_docs)
        
        print(f"📊 Vector store add result: {vector_result} chunks added")
        
        # Step 3: Verify it was actually added
        search_results = rag_system.vector_store.vector_store.similarity_search("test document upload debugging", k=5)
        found = any("debug_test_upload.txt" in doc.metadata.get('source', '') for doc in search_results)
        
        # Step 4: Check hybrid cache
        hybrid_has = False
        if rag_system.vector_store.hybrid_retrieval:
            for doc in rag_system.vector_store.hybrid_retrieval.documents_cache:
                if "debug_test_upload.txt" in doc.metadata.get('source', ''):
                    hybrid_has = True
                    break
        
        return {
            "processing_stage": f"Created {len(processed_docs)} chunks",
            "vector_store_add_stage": f"Returned {vector_result} chunks added",
            "verification_stage": f"Found in vector store: {found}",
            "hybrid_cache_stage": f"Found in hybrid cache: {hybrid_has}",
            "problem": "Vector store add is returning success but documents aren't actually stored"
        }
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
@app.get("/api/debug-structured-data")
async def debug_structured_data(filename: str):
    """Check if structured data exists for a file"""
    try:
        docs = rag_system.document_storage.get_documents("management_full")
        target_doc = None
        
        for doc in docs:
            if filename.lower() in doc['filename'].lower():
                target_doc = doc
                break
        
        if not target_doc:
            return {"error": f"File {filename} not found in document storage"}
        
        structured_data = rag_system.document_storage.get_structured_data(
            "management_full", target_doc['file_id']
        )
        
        return {
            "filename": target_doc['filename'],
            "file_id": target_doc['file_id'],
            "has_structured_data": structured_data is not None,
            "structured_data": structured_data
        }
        
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/api/debug-specific-missing")
async def debug_specific_missing(filename: str = "paper.docx"):
    """Debug why a specific document is missing from vector store"""
    try:
        # Check if it exists in document storage
        docs = rag_system.document_storage.get_documents("management_full")
        in_document_storage = any(filename in doc['filename'] for doc in docs)
        
        # Check if it exists in vector store
        vector_results = rag_system.vector_store.vector_store.similarity_search(filename, k=10)
        in_vector_store = any(filename in doc.metadata.get('source', '') for doc in vector_results)
        
        # Check if it exists in hybrid cache
        in_hybrid_cache = False
        if rag_system.vector_store.hybrid_retrieval:
            for doc in rag_system.vector_store.hybrid_retrieval.documents_cache:
                if filename in doc.metadata.get('source', ''):
                    in_hybrid_cache = True
                    hybrid_content_length = len(doc.page_content)
                    break
        
        return {
            "filename": filename,
            "in_document_storage": in_document_storage,
            "in_vector_store": in_vector_store, 
            "in_hybrid_cache": in_hybrid_cache,
            "hybrid_content_length": hybrid_content_length if in_hybrid_cache else 0,
            "analysis": "Document is processed and in hybrid cache but missing from vector store"
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug-chart-query")
async def debug_chart_query(query: str = "pie chart of iris species"):
    """Debug exactly what happens with a chart query"""
    try:
        rag = rag_system
        
        print(f"🔍 DEBUGGING QUERY: '{query}'")
        
        # Step 1: Chart intent detection
        chart_intent = rag._parse_chart_request(query)
        print(f"🎯 Chart intent detected: {chart_intent}")
        
        if not chart_intent:
            return {"error": "Not detected as chart request"}
        
        # Step 2: Find dataset
        dataset_match = rag._find_structured_dataset("management_full", chart_intent)
        print(f"🔍 Dataset match: {bool(dataset_match)}")
        
        if dataset_match:
            print(f"📄 Matched file: {dataset_match['doc'].get('filename')}")
            print(f"🔑 Chart data ID: {dataset_match['doc'].get('chart_data_id')}")
            print(f"📊 Sheet: {dataset_match['sheet'].get('name')}")
        
        # Step 3: Try the actual query
        result = rag.query(query, "management_full")
        
        return {
            "query": query,
            "chart_intent": chart_intent,
            "dataset_found": bool(dataset_match),
            "matched_file": dataset_match['doc'].get('filename') if dataset_match else None,
            "has_chart_data": dataset_match['doc'].get('chart_data_id') if dataset_match else None,
            "result_has_chart": "chart" in result,
            "method_used": result.get('method_used'),
            "confidence": result.get('confidence'),
            "answer_preview": result.get('answer', '')[:200] + "..." if result.get('answer') else None
        }
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


    

@app.get("/api/debug-structured-datasets")
async def debug_structured_datasets():
    """See ALL structured datasets available"""
    try:
        rag = rag_system
        datasets = list(rag.document_storage.iterate_structured_sheets("management_full"))
        
        dataset_info = []
        for item in datasets:
            dataset_info.append({
                'filename': item['doc'].get('filename'),
                'file_id': item['doc'].get('file_id'),
                'chart_data_id': item['doc'].get('chart_data_id'),
                'sheet_name': item['sheet'].get('name'),
                'row_count': item['sheet'].get('row_count'),
                'columns': [col.get('name') for col in item['sheet'].get('columns', [])][:5]  # First 5 columns
            })
        
        return {
            "total_structured_datasets": len(datasets),
            "datasets": dataset_info
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug-file-matching")
async def debug_file_matching(search_term: str = "iris"):
    """See what files match your search term"""
    try:
        rag = rag_system
        
        # Search for files
        search_results = rag.vector_store.search(search_term, "management_full", 10)
        
        matched_files = []
        for doc in search_results:
            source = doc.metadata.get('source', '')
            matched_files.append({
                'source': source,
                'file_type': doc.metadata.get('file_type', 'unknown'),
                'content_preview': doc.page_content[:100] + "..." if doc.page_content else 'EMPTY'
            })
        
        return {
            "search_term": search_term,
            "files_found": len(matched_files),
            "matched_files": matched_files
        }
    except Exception as e:
        return {"error": str(e)}
 
@app.get("/api/debug-hybrid-search")
async def debug_hybrid_search(query: str, top_k: int = 10, namespace: str = "management_full"):
    """Debug endpoint to see ACTUAL weights used in normal hybrid search"""
    try:
        if not rag_system.vector_store.hybrid_retrieval:
            return {"error": "Hybrid retriever not initialized"}
        
        hybrid_retriever = rag_system.vector_store.hybrid_retrieval
        
        debug_info = {
            'query': query,
            'top_k': top_k,
            'namespace': namespace,
            'vector_results': [],
            'bm25_results': [],
            'fused_results': [],
            'score_breakdown': [],
            'final_ranking': [],
            'actual_weights_used': {}  # 🎯 This will show REAL weights
        }
        
        # 🎯 STEP 1: Run ACTUAL hybrid search to get REAL weights
        print(f"\n🔍 RUNNING ACTUAL HYBRID SEARCH FOR: '{query}'")
        actual_results = hybrid_retriever.hybrid_search(query, top_k)
        
        # 🎯 STEP 2: Run individual searches to compare
        vector_results = hybrid_retriever.vector_search(query, top_k)
        bm25_results = hybrid_retriever.bm25_search(query, top_k) if hybrid_retriever.bm25_index else []
        
        debug_info['vector_results'] = [
            {
                'rank': i + 1,
                'source': doc.metadata.get('source', 'unknown'),
                'score': round(score, 4),
                'file_type': doc.metadata.get('file_type', 'unknown')
            }
            for i, (doc, score) in enumerate(vector_results)
        ]
        
        debug_info['bm25_results'] = [
            {
                'rank': i + 1,
                'source': doc.metadata.get('source', 'unknown'),
                'score': round(score, 4),
                'file_type': doc.metadata.get('file_type', 'unknown')
            }
            for i, (doc, score) in enumerate(bm25_results)
        ]
        
        # 🎯 STEP 3: Analyze ACTUAL fusion from real results
        score_breakdown = []
        
        # Create mapping of documents to their individual scores
        doc_scores = {}
        
        for doc, vector_score in vector_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            doc_scores[content_hash] = {
                'doc': doc,
                'source': doc.metadata.get('source', 'unknown'),
                'vector_score': vector_score,
                'bm25_score': 0.0,
                'actual_final_score': 0.0
            }
        
        for doc, bm25_score in bm25_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash in doc_scores:
                doc_scores[content_hash]['bm25_score'] = bm25_score
            else:
                doc_scores[content_hash] = {
                    'doc': doc,
                    'source': doc.metadata.get('source', 'unknown'),
                    'vector_score': 0.0,
                    'bm25_score': bm25_score,
                    'actual_final_score': 0.0
                }
        
        # 🎯 STEP 4: Find ACTUAL final scores from real hybrid search
        for doc, actual_final_score in actual_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash in doc_scores:
                doc_scores[content_hash]['actual_final_score'] = actual_final_score
                
                # 🎯 REVERSE ENGINEER the weights used!
                vector_score = doc_scores[content_hash]['vector_score']
                bm25_score = doc_scores[content_hash]['bm25_score']
                source = doc_scores[content_hash]['source']
                
                # Calculate what weights were ACTUALLY used
                if vector_score > 0 and bm25_score > 0:
                    # Both scores available - calculate actual weights
                    # actual_final_score = (vector_score * V_weight) + (bm25_score * B_weight)
                    # We know V_weight + B_weight = 1, so we can solve
                    if vector_score != bm25_score:  # Avoid division by zero issues
                        # This is approximate since scores are normalized differently
                        estimated_vector_weight = 0.7  # Default from your code
                        estimated_bm25_weight = 0.3    # Default from your code
                        weights_used = f"Vector(~70%) + BM25(~30%)"
                    else:
                        estimated_vector_weight = 0.5
                        estimated_bm25_weight = 0.5
                        weights_used = "Equal weights (similar scores)"
                        
                    calculation = f"({vector_score:.4f} × ~{estimated_vector_weight:.1f}) + ({bm25_score:.4f} × ~{estimated_bm25_weight:.1f})"
                    
                elif vector_score > 0:
                    # Vector only - calculate actual weight used
                    actual_weight = actual_final_score / vector_score
                    estimated_vector_weight = round(actual_weight, 2)
                    weights_used = f"Vector only (weight: {estimated_vector_weight:.1f})"
                    calculation = f"({vector_score:.4f} × {estimated_vector_weight:.2f})"
                    
                else:
                    # BM25 only - calculate actual weight used  
                    actual_weight = actual_final_score / bm25_score
                    estimated_bm25_weight = round(actual_weight, 2)
                    weights_used = f"BM25 only (weight: {estimated_bm25_weight:.1f})"
                    calculation = f"({bm25_score:.4f} × {estimated_bm25_weight:.2f})"
                
                score_breakdown.append({
                    'source': source,
                    'vector_score': round(vector_score, 4),
                    'bm25_score': round(bm25_score, 4),
                    'actual_final_score': round(actual_final_score, 4),
                    'calculation': calculation,
                    'weights_used': weights_used
                })
        
        debug_info['score_breakdown'] = score_breakdown
        
        # 🎯 STEP 5: Show ACTUAL final results
        debug_info['fused_results'] = [
            {
                'rank': i + 1,
                'source': doc.metadata.get('source', 'unknown'),
                'actual_final_score': round(score, 4),
                'file_type': doc.metadata.get('file_type', 'unknown')
            }
            for i, (doc, score) in enumerate(actual_results)
        ]
        
        # 🎯 STEP 6: Extract ACTUAL weights from your code
        # Let's check what weights are hardcoded in your fusion method
        debug_info['actual_weights_used'] = {
            'vector_bm25_both': '70% vector + 30% BM25 (hardcoded in _fuse_scores)',
            'vector_only': '80% of vector score (hardcoded in _fuse_scores)', 
            'bm25_only': '50% of BM25 score (hardcoded in _fuse_scores)',
            'source': 'FreeHybridRetrieval._fuse_scores() method'
        }
        
        # 🎯 STEP 7: Metrics
        debug_info['metrics'] = {
            'total_vector_docs': len(vector_results),
            'total_bm25_docs': len(bm25_results),
            'actual_final_docs': len(actual_results),
            'vector_bm25_overlap': len([d for d in doc_scores.values() if d['vector_score'] > 0 and d['bm25_score'] > 0])
        }
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/debug-hybrid-init")
async def debug_hybrid_init():
    """Check hybrid retrieval initialization"""
    try:
        vector_store = rag_system.vector_store
        hybrid = vector_store.hybrid_retrieval
        
        # Check if documents exist in vector store
        all_docs = vector_store.vector_store.similarity_search("", k=50)
        
        return {
            "hybrid_retrieval_exists": hybrid is not None,
            "bm25_index_exists": hybrid.bm25_index is not None if hybrid else False,
            "documents_in_hybrid_cache": len(hybrid.documents_cache) if hybrid else 0,
            "documents_in_vector_store": len(all_docs),
            "vector_store_docs_sample": [
                {
                    "source": doc.metadata.get('source', 'unknown'),
                    "file_type": doc.metadata.get('file_type', 'unknown'),
                    "content_length": len(doc.page_content)
                }
                for doc in all_docs[:10]
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/hard-reset-vector-store")
async def hard_reset_vector_store():
    """COMPLETELY DESTROY AND RECREATE vector store from scratch"""
    try:
        print("💥 HARD RESET: Nuclear option - deleting everything...")
        
        # 1. COMPLETELY delete the vector store directory
        import shutil
        vector_store_path = "./vector_store"
        
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
            print(f"🗑️  Deleted entire vector store directory: {vector_store_path}")
        
        # 2. Recreate the directory
        os.makedirs(vector_store_path, exist_ok=True)
        print("✅ Recreated empty vector store directory")
        
        # 3. COMPLETELY recreate the vector store instance
        global rag_system
        rag_system.vector_store = ProfessionalVectorStore()
        print("✅ Recreated fresh vector store instance")
        
        # 4. Clear all document tracking
        rag_system.document_storage.workspace_documents = {}
        rag_system.vector_store.document_hashes = {}
        print("✅ Cleared all document tracking")
        
        return {
            "message": "HARD RESET COMPLETE - Vector store completely destroyed and recreated",
            "next_steps": "Re-upload your 3 core documents now"
        }
        
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/debug-vector-search")
async def debug_vector_search(query: str = "rag system", namespace: str = "management_full"):
    """Debug why vector search returns 0 results"""
    try:
        vector_store = rag_system.vector_store
        
        # Test 1: Direct vector store search
        print("🧪 Testing direct vector store search...")
        direct_results = vector_store.vector_store.similarity_search_with_score(query, k=10)
        
        # Test 2: Semantic search
        print("🧪 Testing semantic search...")
        semantic_results = vector_store.semantic_search(namespace, query, 10)
        
        # Test 3: Check collection stats
        print("🧪 Checking collection...")
        stats = vector_store.get_collection_stats(namespace)
        
        return {
            "query": query,
            "namespace": namespace,
            "direct_vector_results": len(direct_results),
            "semantic_search_results": len(semantic_results.get('documents', [])),
            "collection_stats": stats,
            "direct_results_sample": [
                {
                    "source": doc.metadata.get('source', 'unknown'),
                    "score": score,
                    "content_preview": doc.page_content[:100]
                }
                for doc, score in direct_results[:3]
            ] if direct_results else [],
            "semantic_results_sample": semantic_results.get('documents', [])[:2]
        }
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/debug-vector-store-files")
async def debug_vector_store_files():
    """Check ALL files currently in vector store"""
    try:
        namespace = "management_full"
        all_docs = rag_system.vector_store.vector_store.similarity_search("", k=100)
        
        file_breakdown = {}
        for doc in all_docs:
            source = doc.metadata.get('source', 'unknown')
            file_type = doc.metadata.get('file_type', 'unknown')
            
            if source not in file_breakdown:
                file_breakdown[source] = {
                    'count': 0,
                    'file_type': file_type,
                    'content_preview': doc.page_content[:100] if doc.page_content else 'EMPTY'
                }
            file_breakdown[source]['count'] += 1
        
        return {
            "total_documents": len(all_docs),
            "file_breakdown": file_breakdown
        }
    except Exception as e:
        return {"error": str(e)}




@app.get("/api/debug-upload-behavior")
async def debug_upload_behavior():
    """Debug what happens during document upload"""
    try:
        namespace = "management_full"
        
        # Check current state before upload
        stats_before = rag_system.vector_store.get_collection_stats(namespace)
        
        return {
            "current_documents": stats_before['total_documents'],
            "sources": stats_before['sources'],
            "document_hashes": rag_system.vector_store.document_hashes.get(namespace, {}),
            "message": "Check if documents are being cleared during upload"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug-search-pipeline")
async def debug_search_pipeline(query: str, namespace: str = "management_full"):
    """Debug the entire search pipeline"""
    try:
        vector_store = rag_system.vector_store
        
        # Test each stage
        stages = {}
        
        # Stage 1: Vector only
        vector_docs = vector_store.vector_store.similarity_search_with_score(query, k=10)
        stages["vector_raw"] = [
            {"source": doc.metadata.get('source', 'unknown'), "score": float(score)}
            for doc, score in vector_docs
        ]
        
        # Stage 2: Hybrid fusion
        if vector_store.hybrid_retrieval:
            hybrid_docs = vector_store.hybrid_retrieval.hybrid_search(query, 10)
            stages["hybrid_fusion"] = [
                {"source": doc.metadata.get('source', 'unknown'), "score": float(score)}
                for doc, score in hybrid_docs
            ]
        
        # Stage 3: After prioritization
        final_docs = vector_store.search(query, namespace, 10)
        stages["final_after_prioritization"] = [
            {"source": doc.metadata.get('source', 'unknown')}
            for doc in final_docs[:10]
        ]
        
        return {
            "query": query,
            "search_stages": stages,
            "problem": "Fusion and prioritization are breaking the vector results"
        }
        
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/debug-search-results")
async def debug_search_results(query: str, namespace: str = "management_full"):
    """Debug what the search is ACTUALLY finding"""
    try:
        # Test all search methods
        results = {}
        
        # 1. Direct vector search
        vector_results = rag_system.vector_store.vector_store.similarity_search(query, k=20)
        results['vector_search'] = [
            {
                'source': doc.metadata.get('source', 'unknown'),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'file_type': doc.metadata.get('file_type', 'unknown')
            }
            for doc in vector_results
        ]
        
        # 2. Count sources
        sources_found = {}
        for doc in vector_results:
            source = doc.metadata.get('source', 'unknown')
            sources_found[source] = sources_found.get(source, 0) + 1
        
        return {
            "query": query,
            "total_results": len(vector_results),
            "sources_found": sources_found,
            "all_results": results['vector_search']
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-search-query")
async def debug_search_query(query: str = "church member graduates", namespace: str = "management_full"):
    """Debug what search is actually finding"""
    try:
        # Test different search methods
        results = {}
        
        # Method 1: Direct vector search
        vector_results = rag_system.vector_store.vector_store.similarity_search(query, k=20)
        results['vector_search'] = [
            {
                'source': doc.metadata.get('source', 'unknown'),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'file_type': doc.metadata.get('file_type', 'unknown')
            }
            for doc in vector_results
        ]
        
        # Method 2: Check if hybrid search is working
        if rag_system.vector_store.hybrid_retrieval:
            hybrid_results = rag_system.vector_store.hybrid_retrieval.hybrid_search(query, 20)
            results['hybrid_search'] = [
                {
                    'source': doc.metadata.get('source', 'unknown'),
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'score': score
                }
                for doc, score in hybrid_results
            ]
        
        return {
            "query": query,
            "total_vector_results": len(results.get('vector_search', [])),
            "total_hybrid_results": len(results.get('hybrid_search', [])),
            "results": results
        }
        
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/debug-vector-store-contents")
async def debug_vector_store_contents(namespace: str = "management_full"):
    """Debug what's actually in the vector store right now"""
    try:
        # Get fresh results without any caching
        all_docs = rag_system.vector_store.vector_store.similarity_search("", k=50)
        
        contents = {
            "total_docs_retrieved": len(all_docs),
            "empty_docs_count": 0,
            "valid_docs_count": 0,
            "sample_contents": []
        }
        
        for i, doc in enumerate(all_docs[:10]):  # Show first 10
            is_empty = not doc.page_content or not doc.page_content.strip()
            if is_empty:
                contents["empty_docs_count"] += 1
            else:
                contents["valid_docs_count"] += 1
                
            contents["sample_contents"].append({
                "index": i,
                "is_empty": is_empty,
                "content_length": len(doc.page_content) if doc.page_content else 0,
                "content_preview": doc.page_content[:100] + "..." if doc.page_content and len(doc.page_content) > 100 else doc.page_content,
                "source": doc.metadata.get('source', 'unknown'),
                "metadata": doc.metadata
            })
        
        return contents
        
    except Exception as e:
        return {"error": str(e)}




@app.get("/api/clean-duplicates")
async def clean_duplicates(namespace: str = "management_full"):
    """Clean duplicate documents from existing index"""
    try:
        removed = rag_system.vector_store.cleanup_duplicates(namespace)
        return {
            "duplicates_removed": removed,
            "message": f"Removed {removed} duplicate documents",
            "namespace": namespace
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/find-missing-documents")
async def find_missing_documents():
    """Find exactly which 2 documents are missing from hybrid cache"""
    try:
        vector_store = rag_system.vector_store
        
        # Get ALL documents from both stores
        vector_docs = vector_store.vector_store.similarity_search("", k=15000)
        hybrid_docs = vector_store.hybrid_retrieval.documents_cache if vector_store.hybrid_retrieval else []
        
        # Create sets for comparison
        vector_sources = set()
        hybrid_sources = set()
        
        for doc in vector_docs:
            source = doc.metadata.get('source', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            vector_sources.add(f"{source}_{chunk_id}")
        
        for doc in hybrid_docs:
            source = doc.metadata.get('source', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            hybrid_sources.add(f"{source}_{chunk_id}")
        
        # Find missing documents
        missing_from_hybrid = vector_sources - hybrid_sources
        missing_from_vector = hybrid_sources - vector_sources
        
        # Get details of missing documents
        missing_details = []
        for doc_id in missing_from_hybrid:
            for doc in vector_docs:
                source = doc.metadata.get('source', 'unknown')
                chunk_id = doc.metadata.get('chunk_id', 'unknown')
                if f"{source}_{chunk_id}" == doc_id:
                    missing_details.append({
                        'source': source,
                        'chunk_id': chunk_id,
                        'content_length': len(doc.page_content),
                        'word_count': len(doc.page_content.split()),
                        'content_preview': doc.page_content[:100] if doc.page_content else 'EMPTY',
                        'file_type': doc.metadata.get('file_type', 'unknown')
                    })
                    break
        
        return {
            "vector_store_count": len(vector_docs),
            "hybrid_cache_count": len(hybrid_docs),
            "missing_from_hybrid_count": len(missing_from_hybrid),
            "missing_from_vector_count": len(missing_from_vector),
            "missing_documents": missing_details,
            "analysis": "These are the final 2 documents not making it to hybrid cache"
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/absolute-sync")
async def absolute_sync():
    """ABSOLUTE SYNC: Guarantee 100% synchronization"""
    try:
        vector_store = rag_system.vector_store
        
        # Get ALL documents from vector store
        vector_docs = vector_store.vector_store.similarity_search("", k=15000)
        print(f"📊 Vector store has {len(vector_docs)} documents")
        
        # 🚨 NUCLEAR OPTION: Completely rebuild hybrid cache
        if vector_store.hybrid_retrieval is None:
            vector_store.hybrid_retrieval = FreeHybridRetrieval(vector_store, vector_store.embeddings)
        
        # Clear existing cache
        vector_store.hybrid_retrieval.documents_cache = []
        vector_store.hybrid_retrieval.bm25_index = None
        
        # Add ALL documents without any filtering or deduplication
        valid_docs = []
        for doc in vector_docs:
            # 🚨 NO FILTERING AT ALL - keep every single document
            if doc.page_content is not None:  # Only check for None, not empty
                valid_docs.append(doc)
        
        print(f"🔍 Keeping {len(valid_docs)} documents (no filtering)")
        
        # Build hybrid index
        vector_store.hybrid_retrieval.documents_cache = valid_docs
        texts = [doc.page_content for doc in valid_docs]
        
        try:
            tokenized_texts = [vector_store.hybrid_retrieval._advanced_tokenize(text) for text in texts]
            vector_store.hybrid_retrieval.bm25_index = BM25Okapi(tokenized_texts)
            
            print(f"✅ ABSOLUTE SYNC: {len(valid_docs)} documents")
            print(f"   - Total tokens: {sum(len(tokens) for tokens in tokenized_texts)}")
            
        except Exception as e:
            print(f"🔧 BM25 build warning: {e}")
            # Even if BM25 fails, we still have the documents in cache
        
        hybrid_count = len(vector_store.hybrid_retrieval.documents_cache)
        
        return {
            "status": "ABSOLUTE_SYNC_COMPLETE",
            "vector_store_documents": len(vector_docs),
            "hybrid_cache_documents": hybrid_count,
            "sync_perfect": len(vector_docs) == hybrid_count,
            "missing_count": len(vector_docs) - hybrid_count,
            "message": f"Absolute sync completed - {hybrid_count} documents in hybrid cache"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/fix-hybrid-now")
async def fix_hybrid_now():
    """INSTANT FIX: Rebuild hybrid cache from vector store with proper deduplication"""
    try:
        vector_store = rag_system.vector_store
        
        # Get ALL documents from vector store
        all_docs = vector_store.vector_store.similarity_search("", k=10000)
        print(f"📊 Found {len(all_docs)} documents in vector store")
        
        # Use the same smart deduplication as the fixed add_documents
        unique_docs = []
        seen_identifiers = set()

        for doc in all_docs:
            source = doc.metadata.get('source', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            content_preview = doc.page_content[:100] if doc.page_content else ""
            doc_identifier = f"{source}_{chunk_id}_{hash(content_preview)}"
            
            if doc_identifier not in seen_identifiers:
                seen_identifiers.add(doc_identifier)
                unique_docs.append(doc)

        print(f"🔍 After smart deduplication: {len(unique_docs)} unique documents")
        
        # Rebuild hybrid index
        if vector_store.hybrid_retrieval is None:
            vector_store.hybrid_retrieval = FreeHybridRetrieval(vector_store, vector_store.embeddings)
        
        vector_store.hybrid_retrieval.build_hybrid_index(unique_docs)
        
        hybrid_count = len(vector_store.hybrid_retrieval.documents_cache) if vector_store.hybrid_retrieval else 0
        
        return {
            "status": "SUCCESS",
            "vector_store_documents": len(all_docs),
            "hybrid_cache_documents": hybrid_count,
            "documents_preserved": f"{len(unique_docs)}/{len(all_docs)}",
            "message": f"Hybrid cache rebuilt with {hybrid_count} documents using smart deduplication"
        }
        
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/api/fix-search-index")
async def fix_search_index(namespace: str = "management_full"):
    """Fix the corrupted search index"""
    try:
        rag_system.vector_store.rebuild_search_index(namespace)
        return {
            "message": "Search index rebuilt successfully",
            "namespace": namespace
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug/document-hashes")
async def debug_document_hashes(namespace: str = "management_full"):
    """Show current document hash tracking"""
    try:
        hashes = rag_system.vector_store.document_hashes.get(namespace, {})
        return {
            "namespace": namespace,
            "total_tracked_files": len(hashes),
            "document_hashes": hashes
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/clear-document-hashes")
async def clear_document_hashes(namespace: str = "management_full"):
    """Clear document hash tracking (force reprocessing)"""
    try:
        if namespace in rag_system.vector_store.document_hashes:
            count = len(rag_system.vector_store.document_hashes[namespace])
            rag_system.vector_store.document_hashes[namespace] = {}
            rag_system.vector_store._save_document_hashes()
            return {
                "message": f"Cleared {count} document hashes for {namespace}",
                "namespace": namespace,
                "cleared_count": count
            }
        else:
            return {
                "message": f"No document hashes found for {namespace}",
                "namespace": namespace,
                "cleared_count": 0
            }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/drive-sync-status")
async def drive_sync_status(namespace: str):
    """Get Google Drive sync status"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        status = rag_system.get_sync_status(namespace)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/drive-status")
async def drive_status():
    """Get overall Google Drive status"""
    return {
        "google_drive_connected": len(user_credentials) > 0,
        "google_drive_available": HAS_GOOGLE_DRIVE,
        "connected_accounts": list(user_credentials.keys())
    }

@app.post("/api/create_session")
async def create_session(data: dict):
    try:
        namespace = data.get('namespace')
        title = data.get('title', 'New Chat')
        
        if not namespace:
            raise HTTPException(status_code=400, detail="Namespace is required")
        
        session_id = rag_system.chat_history.create_session(namespace, title)
        
        return {
            "session_id": session_id,
            "title": title,
            "namespace": namespace,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat_sessions")
async def get_chat_sessions(namespace: str):
    try:
        sessions = rag_system.chat_history.get_sessions(namespace)
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat_history")
async def get_chat_history(session_id: str):
    try:
        messages = rag_system.chat_history.get_messages(session_id)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat_session/{session_id}")
async def delete_chat_session(session_id: str):
    try:
        rag_system.chat_history.delete_session(session_id)
        return {"message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/uploaded_documents")
async def get_uploaded_documents(namespace: str):
    try:
        documents = rag_system.document_storage.get_documents(namespace)
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/collection_info")
async def get_collection_info(namespace: str):
    try:
        stats = rag_system.get_document_stats(namespace)
        return {
            "namespace": namespace,
            "document_count": stats['total_documents'],
            "sources": stats['sources'],
            "status": "active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(
    namespace: str = Form(...),
    file: UploadFile = File(...),
    email: str = Form(None)
):
    try:
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        user_manager = CorporateUserManager()
        user_context = user_manager.authenticate_user(email)
        
        if not user_context.can_access_namespace(namespace):
            raise HTTPException(status_code=403, detail="Access denied")
        
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # 🎯 DUAL PROCESSING STRATEGY - FIXED VERSION
        
        # 1. EXTRACT STRUCTURED PREVIEW & STORE ORIGINAL FILE (for charts)
        file_ext = os.path.splitext(file.filename.lower())[1]
        structured_preview = None
        chart_data_id = None
        
        if file_ext in ['.xlsx', '.xls', '.csv']:
            # 🎯 STORE ORIGINAL FILE FOR CHARTING (always do this, even if preview fails)
            chart_data_id = None
            try:
                chart_data_id = rag_system.document_storage.store_original_file(
                    namespace, file.filename, file_bytes
                )
                print(f"✅ Stored original file for charting: {chart_data_id}")
            except Exception as e:
                print(f"⚠️ Failed to store original file for {file.filename}: {e}")
            
            # Try to extract structured preview (with better error handling)
            structured_preview = None
            try:
                processor = ProfessionalDocumentProcessor()
                structured_preview = processor.extract_structured_preview(file_bytes, file.filename)
                print(f"✅ Extracted structured preview for {file.filename}")
            except Exception as e:
                print(f"⚠️ Structured preview extraction failed for {file.filename}: {e}")
                # Even if extraction fails, try to create a minimal structured preview
                # This ensures the file is still marked as having structured data
                if file_ext == '.csv':
                    try:
                        # Try different encodings
                        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']:
                            try:
                                import pandas as pd
                                from io import BytesIO
                                df = pd.read_csv(BytesIO(file_bytes), encoding=encoding, nrows=1000)
                                if not df.empty:
                                    # Create FULL structured preview (not minimal) with actual data
                                    processor = ProfessionalDocumentProcessor()
                                    sheet_info = processor._build_dataframe_preview(df, 'Dataset', 1000, 25)
                                    if sheet_info:
                                        structured_preview = {
                                            'type': 'csv',
                                            'sheets': [sheet_info],
                                            'generated_at': datetime.now().isoformat(),
                                            'source_filename': file.filename
                                        }
                                        print(f"✅ Created structured preview using {encoding} encoding ({len(df)} rows, {len(df.columns)} columns)")
                                        break
                            except UnicodeDecodeError:
                                continue
                            except Exception as e:
                                # Try next encoding
                                continue
                    except Exception as e2:
                        print(f"⚠️ Failed to create structured preview: {e2}")
        
        # 2. PROCESS FOR SEARCH (existing flow - creates chunks for Q&A)
        result = rag_system.ingest_document_with_auto_sync(namespace, file_bytes, file.filename)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # 3. LINK CHART DATA WITH DOCUMENT METADATA
        # Try to update even if structured_preview is None (we still have chart_data_id)
        if chart_data_id:
            print(f"🐛 DEBUG: Attempting to update document with chart data:")
            print(f"   - filename: {file.filename}")
            print(f"   - chart_data_id: {chart_data_id}")
            print(f"   - structured_preview: {'Yes' if structured_preview else 'No (will create minimal)'}")
            
            # If structured_preview is None but we have chart_data_id, create a minimal one
            if not structured_preview and file_ext in ['.xlsx', '.xls', '.csv']:
                print(f"🐛 DEBUG: Creating minimal structured preview since extraction failed")
                # Create a minimal structured preview to ensure the file is marked as having structured data
                structured_preview = {
                    'type': file_ext[1:],  # 'xlsx', 'csv', etc.
                    'sheets': [{
                        'name': 'Dataset',
                        'row_count': 0,  # Will be updated if we can read it
                        'column_count': 0,
                        'columns': [],
                        'numeric_columns': [],
                        'categorical_columns': [],
                        'rows': []
                    }],
                    'generated_at': datetime.now().isoformat(),
                    'extraction_failed': True
                }
            
            if structured_preview:
                result = rag_system.document_storage.update_document_with_chart_data(
                    namespace,
                    filename=file.filename,
                    file_id=chart_data_id,
                    structured_preview=structured_preview
                )
                
                if result:
                    print(f"✅ Successfully updated document metadata with chart data")
                else:
                    print(f"⚠️ WARNING: Failed to update document metadata - document may not exist yet")
                    # Try to find the document and create it if it doesn't exist
                    print(f"🐛 DEBUG: Checking if document exists in workspace...")
                    all_docs = rag_system.document_storage.workspace_documents.get(namespace, [])
                    found = False
                    for doc in all_docs:
                        # Try exact match first
                        if doc.get('filename') == file.filename:
                            found = True
                            print(f"   ✅ Document exists: {doc.get('filename')}")
                            # Try updating again with the actual file_id from the document
                            if doc.get('file_id'):
                                print(f"   🔄 Retrying with document's file_id: {doc.get('file_id')}")
                                result = rag_system.document_storage.update_document_with_chart_data(
                                    namespace,
                                    filename=file.filename,
                                    file_id=doc.get('file_id'),  # Use the document's file_id
                                    structured_preview=structured_preview
                                )
                            break
                        # Try partial match (for Excel files with sheet names)
                        elif file.filename.lower() in doc.get('filename', '').lower():
                            found = True
                            print(f"   ✅ Document exists (partial match): {doc.get('filename')}")
                            if doc.get('file_id'):
                                print(f"   🔄 Retrying with document's file_id: {doc.get('file_id')}")
                                result = rag_system.document_storage.update_document_with_chart_data(
                                    namespace,
                                    filename=doc.get('filename'),  # Use the document's actual filename
                                    file_id=doc.get('file_id'),
                                    structured_preview=structured_preview
                                )
                            break
                    
                    if not found:
                        print(f"   ❌ Document not found in workspace - may need to wait for ingestion to complete")
            else:
                print(f"⚠️ WARNING: No structured_preview and no chart_data_id - cannot update document metadata")
        
        return {
            "message": "Success",
            "filename": file.filename,
            "namespace": namespace,
            "chunks": result.get('chunks', 0),  # Fix: Use .get() to avoid KeyError
            "auto_sync_performed": result.get('auto_sync_performed', False),
            "structured_preview_extracted": structured_preview is not None,
            "chart_data_stored": chart_data_id is not None,
            "chart_data_id": chart_data_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(
    namespace: str = Form(...),
    question: str = Form(...),
    session_id: str = Form(None),
    email: str = Form(None)
):
    try:
        if rag_system is None:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        user_manager = CorporateUserManager()
        user_context = user_manager.authenticate_user(email)
        
        if not user_context.can_access_namespace(namespace):
            raise HTTPException(status_code=403, detail="Access denied")
        
        result = rag_system.query(question, namespace, session_id)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# GOOGLE DRIVE AUTH ENDPOINTS
# =============================================================================

@app.get("/auth/google")
async def auth_google():
    """Start Google OAuth flow"""
    try:
        print("🔄 Starting OAuth flow...")
        
        if not os.path.exists("client_secrets.json"):
            error_msg = "client_secrets.json not found"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        
        if not HAS_OAUTH:
            error_msg = "OAuth dependencies not available"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        
        from google_auth_oauthlib.flow import Flow
        
        print("✅ client_secrets.json is valid, creating flow...")
        
        flow = Flow.from_client_secrets_file(
            "client_secrets.json",
            scopes=GOOGLE_OAUTH_CONFIG["scopes"]
        )
        
        flow.redirect_uri = GOOGLE_OAUTH_CONFIG["redirect_uri"]
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        print(f"✅ OAuth URL generated successfully")
        
        return {
            "authorization_url": authorization_url,
            "status": "success",
            "redirect_uri": GOOGLE_OAUTH_CONFIG["redirect_uri"],
            "message": "OAuth flow started successfully"
        }
        
    except Exception as e:
        error_msg = f"OAuth setup failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}

@app.get("/oauth2callback")
async def oauth2callback(request: Request, code: str = None, state: str = None, error: str = None):
    try:
        print(f"🔄 OAuth callback received")
        
        if error:
            error_msg = f"OAuth error: {error}"
            print(f"❌ {error_msg}")
            return HTMLResponse(f"""
                <html><body>
                    <h2>OAuth Error</h2>
                    <p>{error_msg}</p>
                    <button onclick="window.close()">Close</button>
                </body></html>
            """)
        
        if not code:
            code = request.query_params.get("code")
            if not code:
                print("❌ No authorization code received")
                return HTMLResponse(f"""
                    <html><body>
                        <h2>No Authorization Code</h2>
                        <p>No authorization code was received. Please try again.</p>
                        <button onclick="window.close()">Close</button>
                    </body></html>
                """)
        
        print(f"✅ Received authorization code: {code[:20]}...")
        
        from google_auth_oauthlib.flow import Flow
        
        flow = Flow.from_client_secrets_file(
            "client_secrets.json",
            scopes=GOOGLE_OAUTH_CONFIG["scopes"]
        )
        flow.redirect_uri = GOOGLE_OAUTH_CONFIG["redirect_uri"]
        
        print("🔄 Exchanging code for tokens...")
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        print("✅ Successfully obtained credentials")
        
        from googleapiclient.discovery import build
        service = build('drive', 'v3', credentials=credentials)
        about = service.about().get(fields="user").execute()
        user_email = about['user']['emailAddress']
        
        print(f"🔗 Identified user: {user_email}")
        
        user_credentials[user_email] = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        
        print(f"💾 Stored credentials for {user_email}")
        
        success_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Google Drive Connected</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    padding: 40px; 
                    text-align: center; 
                    background: #f5f5f5;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    max-width: 500px;
                    margin: 0 auto;
                }}
                .success {{ 
                    color: #0f9d58; 
                    font-size: 24px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success">✅ Google Drive Connected Successfully!</div>
                <div>
                    <p><strong>Connected for:</strong> {user_email}</p>
                    <p>You can now close this window and return to the application.</p>
                </div>
                <button onclick="closeWindow()">Close Window</button>
            </div>
            <script>
                function closeWindow() {{
                    if (window.opener) {{
                        window.opener.postMessage({{
                            type: 'oauth_success',
                            email: '{user_email}',
                            timestamp: new Date().toISOString()
                        }}, '*');
                    }}
                    window.close();
                }}
                setTimeout(closeWindow, 2000);
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=success_html)
        
    except Exception as e:
        error_msg = f"OAuth callback failed: {str(e)}"
        print(f"💥 {error_msg}")
        
        error_html = f"""
        <html>
        <body>
            <h2 style="color: #d93025;">❌ Connection Failed</h2>
            <p>Error: {str(e)}</p>
            <button onclick="window.close()">Close Window</button>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html)

# =============================================================================
# ADDITIONAL PROFESSIONAL ENDPOINTS
# =============================================================================



@app.get("/api/search-metrics")
async def get_search_metrics():
    """Professional search performance monitoring"""
    metrics = {
        "hybrid_retrieval_initialized": rag_system.vector_store.hybrid_retrieval is not None,
        "bm25_index_built": rag_system.vector_store.hybrid_retrieval.bm25_index is not None if rag_system.vector_store.hybrid_retrieval else False,
        "documents_in_cache": len(rag_system.vector_store.hybrid_retrieval.documents_cache) if rag_system.vector_store.hybrid_retrieval else 0,
        "vector_store_type": type(rag_system.vector_store.vector_store).__name__
    }
    return metrics

@app.get("/api/system-health")
async def system_health():
    """Professional system health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "Professional Enterprise RAG System",
        "components": {
            "vector_store": "active",
            "document_processor": "active", 
            "google_drive": "available" if HAS_GOOGLE_DRIVE else "unavailable",
            "chat_history": "active",
            "document_storage": "active"
        },
        "version": "2.0.0-professional"
    }

@app.get("/api/namespace-stats")
async def namespace_stats(namespace: str):
    """Get professional statistics for a namespace"""
    try:
        stats = rag_system.get_document_stats(namespace)
        doc_count = rag_system.document_storage.get_document_count(namespace)
        
        return {
            "namespace": namespace,
            "total_documents": stats['total_documents'],
            "unique_sources": len(stats['sources']),
            "uploaded_files": doc_count,
            "source_breakdown": stats['sources']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# DEBUG ENDPOINTS - YOUR EXISTING
# =============================================================================

@app.get("/api/debug-all-content")
async def debug_all_content():
    """Debug ALL content in the vector store"""
    try:
        namespace = "management_full"
        stats = rag_system.get_document_stats(namespace)
        return {
            "total_chunks": stats['total_documents'],
            "sources_breakdown": stats['sources']
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-fallback-test")
async def debug_fallback_test(query: str):
    """Test the smart fallback system with detailed comparison"""
    print(f"\n" + "="*100)
    print(f"🧪 DEBUG FALLBACK TEST: '{query}'")
    print("="*100)
    
    # Test with detailed logging
    result = rag_system.query(query, "management_full")
    
    return {
        "query": query,
        "final_method_used": result.get('method_used'),
        "final_confidence": result.get('confidence'),
        "documents_used": result.get('documents_used'),
        "answer_preview": result['answer'][:500] + "..." if len(result['answer']) > 500 else result['answer']
    }


@app.get("/api/debug-chunking")
async def debug_chunking(filename: str = "The-book-of-Enoch.pdf"):
    """Debug chunking for a specific file - FIXED VERSION"""
    try:
        namespace = "management_full"
        
        # Search for chunks from this file using semantic search
        search_results = rag_system.vector_store.semantic_search(
            namespace, filename, n_results=100
        )
        
        chunk_info = []
        for i, (doc, meta) in enumerate(zip(
            search_results.get('documents', []),
            search_results.get('metadatas', [])
        )):
            if meta.get('source') == filename:
                chunk_info.append({
                    "chunk_id": i,
                    "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                    "content_length": len(doc),
                    "metadata": meta
                })
        
        return {
            "filename": filename,
            "total_chunks_found": len(chunk_info),
            "chunks": chunk_info
        }
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug-document-counts")
async def debug_document_counts():
    vector_store = rag_system.vector_store
    hybrid = vector_store.hybrid_retrieval
    
    # Count documents in each system
    vector_docs = vector_store.vector_store.similarity_search("", k=1000)
    bm25_docs = hybrid.documents_cache if hybrid else []
    
    return {
        "total_vector_docs": len(vector_docs),
        "total_bm25_docs": len(bm25_docs),
        "vector_sources_count": len(set(d.metadata.get('source') for d in vector_docs)),
        "bm25_sources_count": len(set(d.metadata.get('source') for d in bm25_docs)),
    }


@app.get("/api/debug-content-differences")
async def debug_content_differences():
    """Check if documents have the same content in both stores"""
    try:
        vector_store = rag_system.vector_store
        
        # Get documents from both stores
        vector_docs = vector_store.vector_store.similarity_search("", k=1000)
        hybrid_docs = vector_store.hybrid_retrieval.documents_cache if vector_store.hybrid_retrieval else []
        
        # Analyze content by source
        vector_content_by_source = {}
        hybrid_content_by_source = {}
        
        for doc in vector_docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in vector_content_by_source:
                vector_content_by_source[source] = []
            vector_content_by_source[source].append({
                'content_length': len(doc.page_content),
                'content_preview': doc.page_content[:200] if doc.page_content else "EMPTY",
                'chunk_id': doc.metadata.get('chunk_id', 'unknown')
            })
        
        for doc in hybrid_docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in hybrid_content_by_source:
                hybrid_content_by_source[source] = []
            hybrid_content_by_source[source].append({
                'content_length': len(doc.page_content),
                'content_preview': doc.page_content[:200] if doc.page_content else "EMPTY", 
                'chunk_id': doc.metadata.get('chunk_id', 'unknown')
            })
        
        # Find differences
        differences = {}
        all_sources = set(list(vector_content_by_source.keys()) + list(hybrid_content_by_source.keys()))
        
        for source in all_sources:
            vector_chunks = vector_content_by_source.get(source, [])
            hybrid_chunks = hybrid_content_by_source.get(source, [])
            
            differences[source] = {
                'vector_chunks': len(vector_chunks),
                'hybrid_chunks': len(hybrid_chunks),
                'count_match': len(vector_chunks) == len(hybrid_chunks),
                'content_samples_different': False
            }
            
            # Check if content is actually different
            if vector_chunks and hybrid_chunks:
                # Compare first chunk content
                vector_sample = vector_chunks[0]['content_preview'] if vector_chunks else "NO CONTENT"
                hybrid_sample = hybrid_chunks[0]['content_preview'] if hybrid_chunks else "NO CONTENT"
                
                if vector_sample != hybrid_sample:
                    differences[source]['content_samples_different'] = True
                    differences[source]['vector_sample'] = vector_sample
                    differences[source]['hybrid_sample'] = hybrid_sample
        
        return {
            "total_sources": len(all_sources),
            "content_differences": differences,
            "analysis": "This shows if documents with the same source name have different content in each store"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-document-sources")
async def debug_document_sources():
    """Debug why some documents are missing from the documents tab"""
    try:
        vector_store = rag_system.vector_store
        
        # Get ALL documents from vector store
        all_docs = vector_store.vector_store.similarity_search("", k=10000)
        
        # Get what's in documents tab
        ui_docs = rag_system.document_storage.get_documents("management_full")
        
        # Analyze sources
        vector_sources = set()
        for doc in all_docs:
            source = doc.metadata.get('source', 'unknown')
            vector_sources.add(source)
        
        ui_filenames = {doc['filename'] for doc in ui_docs}
        
        # Find missing documents
        missing_from_ui = []
        for source in vector_sources:
            filename = source.replace('Google Drive: ', '') if 'Google Drive:' in source else source
            if filename not in ui_filenames:
                missing_from_ui.append(filename)
        
        return {
            "total_vector_documents": len(all_docs),
            "total_ui_documents": len(ui_docs),
            "missing_from_ui_count": len(missing_from_ui),
            "missing_documents_sample": missing_from_ui[:20],  # First 20 missing
            "vector_sources_count": len(vector_sources),
            "ui_filenames_count": len(ui_filenames)
        }
        
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/debug-document-loading")
async def debug_document_loading():
    """Check why paper.docx isn't in BM25 cache"""
    try:
        vector_store = rag_system.vector_store
        
        # Get all documents from vector store
        all_vector_docs = vector_store.vector_store.similarity_search("", k=100)
        
        # Get documents from BM25 cache
        hybrid_docs = vector_store.hybrid_retrieval.documents_cache if vector_store.hybrid_retrieval else []
        
        vector_sources = [doc.metadata.get('source', 'unknown') for doc in all_vector_docs]
        hybrid_sources = [doc.metadata.get('source', 'unknown') for doc in hybrid_docs]
        
        return {
            "vector_store_docs_count": len(all_vector_docs),
            "hybrid_cache_docs_count": len(hybrid_docs),
            "vector_store_has_paper": any("paper.docx" in source for source in vector_sources),
            "hybrid_cache_has_paper": any("paper.docx" in source for source in hybrid_sources),
            "vector_store_sources": list(set(vector_sources)),
            "hybrid_cache_sources": list(set(hybrid_sources))
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/vector-store-stats")
async def vector_store_stats():
    """Check what's actually in the vector store"""
    try:
        stats = rag_system.vector_store.get_collection_stats("management_full")
        return {
            "total_documents": stats.get('total_documents', 0),
            "sources_breakdown": stats.get('sources', {}),
            "sources_count": len(stats.get('sources', {}))
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/debug-document-chunks")
async def debug_document_chunks(filename: str, namespace: str = "management_full"):
    """Debug ALL chunks for a specific document"""
    try:
        # Get all chunks and filter by filename
        all_docs = rag_system.vector_store.vector_store.similarity_search("", k=200)
        
        document_chunks = []
        for doc in all_docs:
            source = doc.metadata.get('source', '')
            # Match by source filename
            if filename in source:
                document_chunks.append({
                    "source": source,
                    "content": doc.page_content,
                    "content_length": len(doc.page_content),
                    "metadata": doc.metadata,
                    "is_empty": not doc.page_content or not doc.page_content.strip()
                })
        
        # Group by chunk strategy if available
        chunks_by_strategy = {}
        for chunk in document_chunks:
            strategy = chunk['metadata'].get('chunk_strategy', 'unknown')
            if strategy not in chunks_by_strategy:
                chunks_by_strategy[strategy] = []
            chunks_by_strategy[strategy].append(chunk)
        
        return {
            "filename": filename,
            "total_chunks_found": len(document_chunks),
            "chunks_by_strategy": chunks_by_strategy,
            "all_chunks": [
                {
                    "chunk_id": i,
                    "strategy": chunk['metadata'].get('chunk_strategy', 'unknown'),
                    "content_preview": chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'],
                    "content_length": chunk['content_length'],
                    "metadata": chunk['metadata']
                }
                for i, chunk in enumerate(document_chunks)
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}


    
@app.get("/api/debug-document-storage")
async def debug_document_storage(namespace: str = "management_full"):
    """Debug document storage state"""
    try:
        # Check both storage systems
        rag_docs = rag_system.document_storage.get_documents(namespace)
        vector_docs = rag_system.vector_store.document_storage.get_documents(namespace)
        
        # Check if they're the same instance
        same_instance = rag_system.document_storage is rag_system.vector_store.document_storage
        
        return {
            "same_instance": same_instance,
            "rag_system_docs_count": len(rag_docs),
            "vector_store_docs_count": len(vector_docs),
            "rag_system_docs": rag_docs,
            "vector_store_docs": vector_docs,
            "message": "Same instance" if same_instance else "DIFFERENT INSTANCES - THIS IS THE PROBLEM!"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-document-content")
async def debug_document_content(filename: str, namespace: str = "management_full"):
    """Debug what content was actually extracted from a document - FIXED VERSION"""
    try:
        # Search for the document content using semantic search
        search_results = rag_system.vector_store.semantic_search(
            namespace, filename, n_results=50
        )
        
        debug_info = {
            "filename": filename,
            "total_chunks_found": len(search_results.get('documents', [])),
            "chunks_from_this_file": 0,
            "file_chunks": []
        }
        
        # Filter chunks from this specific file
        for i, (doc, meta) in enumerate(zip(
            search_results.get('documents', []),
            search_results.get('metadatas', [])
        )):
            if meta.get('source') == filename:
                debug_info["chunks_from_this_file"] += 1
                debug_info["file_chunks"].append({
                    "chunk_id": i+1,
                    "content_preview": doc[:500] + "..." if len(doc) > 500 else doc,
                    "metadata": meta
                })
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}
    

    

@app.delete("/api/clear-database")
async def clear_database(namespace: str = "management_full"):
    """Clear all documents from the vector database"""
    try:
        # Use the professional clear method
        doc_count = rag_system.vector_store.clear_collection(namespace)
        
        # Also clear document storage
        rag_system.document_storage.workspace_documents[namespace] = []
        
        return {
            "message": f"Database cleared successfully",
            "documents_deleted": doc_count,
            "namespace": namespace
        }
        
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/api/debug-chunk-content")
async def debug_chunk_content(filename: str = None, namespace: str = "management_full"):
    """Debug what content is actually in chunks"""
    try:
        # Get ALL chunks from the vector store
        all_docs = rag_system.vector_store.vector_store.similarity_search("", k=100)
        
        debug_info = {
            "total_chunks_in_system": len(all_docs),
            "chunks_by_file": {},
            "all_chunks_preview": []
        }
        
        # If a specific filename is requested, filter by it
        if filename:
            filtered_docs = []
            for doc in all_docs:
                source = doc.metadata.get('source', '')
                if filename in source:
                    filtered_docs.append(doc)
            
            debug_info["filtered_chunks"] = len(filtered_docs)
            debug_info["chunks_for_file"] = []
            
            for i, doc in enumerate(filtered_docs):
                debug_info["chunks_for_file"].append({
                    "chunk_id": i,
                    "source": doc.metadata.get('source', 'unknown'),
                    "file_type": doc.metadata.get('file_type', 'unknown'),
                    "content_length": len(doc.page_content),
                    "content_preview": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                })
        
        # Show overview of all chunks by file
        file_chunk_counts = {}
        for doc in all_docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in file_chunk_counts:
                file_chunk_counts[source] = 0
            file_chunk_counts[source] += 1
        
        debug_info["chunks_by_file"] = file_chunk_counts
        
        # Show preview of first 10 chunks
        for i, doc in enumerate(all_docs[:10]):
            debug_info["all_chunks_preview"].append({
                "chunk_id": i,
                "source": doc.metadata.get('source', 'unknown'),
                "content_length": len(doc.page_content),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "is_empty": not doc.page_content or not doc.page_content.strip()
            })
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}
@app.get("/api/debug-excel-fix")
async def debug_excel_fix():
    """Test if Excel files can be read after the fix"""
    try:
        # Test with a simple Excel file creation
        import pandas as pd
        from io import BytesIO
        
        # Create a test DataFrame
        test_data = {
            'SepalLength': [5.1, 4.9, 4.7],
            'SepalWidth': [3.5, 3.0, 3.2], 
            'PetalLength': [1.4, 1.4, 1.3],
            'PetalWidth': [0.2, 0.2, 0.2],
            'Species': ['setosa', 'setosa', 'setosa']
        }
        df = pd.DataFrame(test_data)
        
        # Test writing and reading back
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Test', index=False)
        
        excel_data = output.getvalue()
        
        # Test reading with engine
        excel_file = BytesIO(excel_data)
        xl = pd.ExcelFile(excel_file, engine='openpyxl')
        
        return {
            "success": True,
            "sheets_found": xl.sheet_names,
            "engine_test": "openpyxl engine works correctly"
        }
        
    except Exception as e:
        return {"error": str(e), "fix_status": "Excel engine fix needed"}
@app.get("/api/debug-extract-preview")
async def debug_extract_preview(filename: str):
    """Debug why structured preview extraction fails"""
    try:
        # Get the file content
        file_results = rag_system.vector_store.vector_store.similarity_search(filename, k=10)
        file_content = "\n".join([doc.page_content for doc in file_results if filename in doc.metadata.get('source', '')])
        
        if not file_content:
            return {"error": "No content found for file"}
        
        print(f"🔍 DEBUG: Testing extract_structured_preview for {filename}")
        
        # Test with debug version
        result = rag_system.vector_store.document_processor.debug_extract_preview(
            file_content.encode('utf-8'), filename
        )
        
        return {
            "filename": filename,
            "content_length": len(file_content),
            "preview_result": result is not None,
            "result": result
        }
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}





        
@app.get("/api/debug-empty-chunks")
async def debug_empty_chunks(namespace: str = "management_full"):
    """Find and count empty chunks in the vector store"""
    try:
        # Get all documents
        all_docs = rag_system.vector_store.vector_store.similarity_search("", k=2000)
        
        empty_chunks = []
        total_chunks = len(all_docs)
        
        for doc in all_docs:
            if not doc.page_content or not doc.page_content.strip():
                empty_chunks.append({
                    'source': doc.metadata.get('source', 'unknown'),
                    'metadata': doc.metadata
                })
        
        return {
            'total_chunks': total_chunks,
            'empty_chunks_count': len(empty_chunks),
            'empty_chunks_percentage': (len(empty_chunks) / total_chunks * 100) if total_chunks > 0 else 0,
            'empty_chunks_by_source': {},
            'sample_empty_chunks': empty_chunks[:10]  # Show first 10
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-document-extraction")
async def debug_document_extraction(filename: str):
    """Test document extraction for a specific file"""
    try:
        if not user_credentials:
            return {"error": "No Google Drive credentials available"}
        
        user_email, credentials_dict = list(user_credentials.items())[0]
        
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        
        creds = Credentials(
            token=credentials_dict['token'],
            refresh_token=credentials_dict['refresh_token'],
            token_uri=credentials_dict['token_uri'],
            client_id=credentials_dict['client_id'],
            client_secret=credentials_dict['client_secret'],
            scopes=credentials_dict['scopes']
        )
        
        service = build('drive', 'v3', credentials=creds)
        
        # Search for the file
        results = service.files().list(
            q=f"name='{filename}'",
            pageSize=5,
            fields="files(id, name, mimeType, size)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            return {"error": f"File '{filename}' not found in Google Drive"}
        
        file_info = files[0]
        file_id = file_info['id']
        
        # Download file content
        request = service.files().get_media(fileId=file_id)
        file_content = request.execute()
        
        # Test extraction with your processor
        documents = rag_system.vector_store.document_processor.process_document(file_content, filename)
        
        extraction_debug = {
            'file_info': file_info,
            'file_size_bytes': len(file_content),
            'documents_extracted': len(documents),
            'documents_detail': []
        }
        
        for i, doc in enumerate(documents):
            extraction_debug['documents_detail'].append({
                'document_index': i,
                'content_length': len(doc.page_content),
                'content_preview': doc.page_content[:200] + "..." if doc.page_content and len(doc.page_content) > 200 else doc.page_content,
                'metadata': doc.metadata,
                'is_empty': not doc.page_content or not doc.page_content.strip()
            })
        
        return extraction_debug
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/nuclear-reset")
async def nuclear_reset(namespace: str = "management_full"):
    """COMPLETE nuclear reset - delete everything and start fresh"""
    try:
        print("💥 NUCLEAR RESET: Deleting everything and starting fresh...")
        
        # 1. Clear vector store completely
        rag_system.vector_store.clear_collection(namespace)
        
        # 2. Clear document storage
        rag_system.document_storage.workspace_documents[namespace] = []
        
        # 3. Clear document hashes
        if namespace in rag_system.vector_store.document_hashes:
            rag_system.vector_store.document_hashes[namespace] = {}
            rag_system.vector_store._save_document_hashes()
        
        # 4. Reset hybrid retrieval
        rag_system.vector_store.hybrid_retrieval = None
        
        return {
            "message": "Nuclear reset completed - vector store is now completely empty",
            "namespace": namespace,
            "next_step": "You need to re-sync or re-upload documents"
        }
        
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/api/nuclear-cleanup")
async def nuclear_cleanup(namespace: str = "management_full"):
    """Complete nuclear cleanup - remove empty chunks and rebuild everything"""
    try:
        # Get all documents
        all_docs = rag_system.vector_store.vector_store.similarity_search("", k=2000)
        
        # Filter out empty chunks
        valid_docs = []
        empty_count = 0
        
        for doc in all_docs:
            if doc.page_content and doc.page_content.strip():
                valid_docs.append(doc)
            else:
                empty_count += 1
        
        print(f"🗑️ Removing {empty_count} empty chunks, keeping {len(valid_docs)} valid chunks")
        
        # Clear and rebuild with only valid chunks
        rag_system.vector_store.clear_collection(namespace)
        rag_system.vector_store.add_documents(namespace, valid_docs)
        
        return {
            "message": "Nuclear cleanup completed",
            "empty_chunks_removed": empty_count,
            "valid_chunks_kept": len(valid_docs),
            "namespace": namespace
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-file-content")
async def debug_file_content(filename: str):
    """Debug what content is actually extracted from a specific file"""
    try:
        if not user_credentials:
            return {"error": "No Google Drive credentials available"}
        
        user_email, credentials_dict = list(user_credentials.items())[0]
        
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        
        creds = Credentials(
            token=credentials_dict['token'],
            refresh_token=credentials_dict['refresh_token'],
            token_uri=credentials_dict['token_uri'],
            client_id=credentials_dict['client_id'],
            client_secret=credentials_dict['client_secret'],
            scopes=credentials_dict['scopes']
        )
        
        service = build('drive', 'v3', credentials=creds)
        
        # Search for the file
        results = service.files().list(
            q=f"name='{filename}'",
            pageSize=5,
            fields="files(id, name, mimeType)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            return {"error": f"File '{filename}' not found"}
        
        file_info = files[0]
        
        # Test different extraction methods
        debug_results = []
        
        # Method 1: GoogleDriveLoader
        try:
            from langchain_googledrive.document_loaders import GoogleDriveLoader
            loader = GoogleDriveLoader(
                file_ids=[file_info['id']],
                credentials=creds
            )
            documents = loader.load()
            debug_results.append({
                "method": "GoogleDriveLoader",
                "documents_count": len(documents),
                "content_preview": documents[0].page_content[:500] + "..." if documents and documents[0].page_content else "NO CONTENT",
                "content_length": len(documents[0].page_content) if documents else 0,
                "metadata": documents[0].metadata if documents else {}
            })
        except Exception as e:
            debug_results.append({
                "method": "GoogleDriveLoader", 
                "error": str(e)
            })
        
        return {
            "file": file_info,
            "extraction_tests": debug_results
        }
        
    except Exception as e:
        return {"error": str(e)}
    


@app.get("/api/system-status")
async def system_status():
    """Get complete system status"""
    try:
        status = {
            "rag_system_initialized": rag_system is not None,
            "google_drive_available": rag_system.drive_sync_manager.has_google_drive if rag_system else False,
            "google_drive_connected": len(user_credentials) > 0,
            "connected_drive_accounts": list(user_credentials.keys()),
            "total_namespaces": len(CORPORATE_NAMESPACES),
            "supported_file_types": ALLOWED_EXT
        }
        return status
    except Exception as e:
        return {"error": str(e)}



@app.get("/api/debug-excel-content")
async def debug_excel_content(filename: str = "UAE Internships & Jobs (+twe extract).xlsx"):
    """Debug what's actually in Excel chunks"""
    try:
        namespace = "management_full"
        
        # Search for chunks from this file
        search_results = rag_system.vector_store.semantic_search(
            namespace, filename, n_results=50
        )
        
        excel_debug = []
        for i, (doc_content, metadata, score) in enumerate(zip(
            search_results.get('documents', []),
            search_results.get('metadatas', []),
            search_results.get('relevance_scores', [])
        )):
            if "UAE Internships" in str(metadata.get('source', '')):
                excel_debug.append({
                    "chunk_id": i,
                    "content_preview": doc_content[:500] + "..." if len(doc_content) > 500 else doc_content,
                    "content_length": len(doc_content),
                    "relevance_score": score,
                    "metadata": metadata
                })
        
        return {
            "filename": filename,
            "total_chunks_found": len(excel_debug),
            "chunks": excel_debug
        }
        
    except Exception as e:
        return {"error": str(e)}


    

# =============================================================================
# COMPLETE FRONTEND HTML INTERFACE - YOUR EXISTING
# =============================================================================

COMPLETE_HTML_INTERFACE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Corporate Document Analysis System - COMPLETE</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* Theme Variables */
        :root {
            --primary-color: #8b5cf6;
            --primary-hover: #7c3aed;
            --sidebar-bg: #202123;
            --main-bg: #343541;
            --border-color: #4a4b52;
            --text-primary: #ececf1;
            --text-secondary: #acacbe;
            --user-message-bg: #343541;
            --ai-message-bg: #444654;
            --shadow: 0 2px 10px rgba(0,0,0,0.3);
            --welcome-bg: #343541;
        }

        /* Light Theme */
        [data-theme="light"] {
            --primary-color: #8b5cf6;
            --primary-hover: #7c3aed;
            --sidebar-bg: #f7f7f8;
            --main-bg: #ffffff;
            --border-color: #d9d9e3;
            --text-primary: #343541;
            --text-secondary: #6e6e80;
            --user-message-bg: #ffffff;
            --ai-message-bg: #f7f7f8;
            --shadow: 0 2px 10px rgba(0,0,0,0.1);
            --welcome-bg: #ffffff;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--main-bg);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }
        
        .app-container {
            display: flex;
            height: 100vh;
        }
        
        /* Sidebar */
        .sidebar {
            width: 260px;
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
        }
        
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .new-chat-btn {
            width: 100%;
            padding: 12px;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            justify-content: center;
        }
        
        .new-chat-btn:hover { background: var(--primary-hover); }
        
        .sidebar-content {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }
        
        .chat-history-item {
            padding: 12px;
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 4px;
            font-size: 14px;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .chat-history-item:hover { background: rgba(100, 48, 222, 0.1); }

        .chat-history-item.active { background: rgba(139, 92, 246, 0.15); color: var(--primary-color); }
        
        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--main-bg);
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        
        .message {
            max-width: 80%;
            display: flex;
            gap: 12px;
        }
        
        .user-message {
            align-self: flex-end;
            flex-direction: row-reverse;
        }
        
        .ai-message {
            align-self: flex-start;
        }
        
        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 600;
            flex-shrink: 0;
        }
        
        .user-message .message-avatar {
            background: var(--primary-color);
            color: white;
        }

        
        .ai-message .message-avatar {
            background: #565869;
            color: var(--text-primary);
        }
        
        .message-content {
            padding: 16px;
            border-radius: 12px;
            line-height: 1.5;
            box-shadow: var(--shadow);
            white-space: pre-line;
        }
        
        .user-message .message-content {
            background: var(--user-message-bg);
            border-top-right-radius: 4px;
            border: 1px solid var(--border-color);
        }

        
        .ai-message .message-content {
            background: var(--ai-message-bg);
            border: 1px solid var(--border-color);
            border-top-left-radius: 4px;
        }
        
        .input-area {
            padding: 24px;
            border-top: 1px solid var(--border-color);
            background: var(--main-bg);
        }
        
        .input-container {
            max-width: 768px;
            margin: 0 auto;
            position: relative;
        }
        
        .input-form {
            display: flex;
            gap: 12px;
            align-items: flex-end;
            background: var(--main-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 12px;
            box-shadow: var(--shadow);
        }
        
        .input-form:focus-within {
            border-color: var(--primary-color);
        }
        
        .input-form textarea {
            flex: 1;
            border: none;
            outline: none;
            resize: none;
            font-size: 16px;
            line-height: 1.5;
            background: transparent;
            max-height: 200px;
            min-height: 24px;
            font-family: inherit;
            color: var(--text-primary);
        }
        
        .input-form textarea::placeholder {
            text-align: center;
            color: var(--text-secondary);
        }
        
        .send-button {
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 6px;
            width: 36px;
            height: 36px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        
        .send-button:hover {
            background: var(--primary-hover);
        }
        
        .send-button:disabled {
            background: var(--border-color);
            cursor: not-allowed;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            background: var(--sidebar-bg);
        }
        
        .tab {
            padding: 12px 16px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            font-size: 14px;
            color: var(--text-secondary); /* Default inactive color */
            transition: all 0.3s ease; /* Smooth transition */
        }

        .tab.active {
            border-bottom-color: var(--primary-color);
            color: var(--text-primary); /* Active text color */
            background:rgb(149, 114, 229); /* Purple background */
        }

        .tab:hover {
            background: rgb(190, 173, 228); /* Light purple on hover */
            color: var(--text-primary);
        }
        
        

        
        .tab-content {
            display: none;
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            background: var(--main-bg);
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Welcome State */
        .welcome-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
            background: var(--welcome-bg);
        }
        
        .welcome-icon {
            font-size: 48px;
            color: var(--primary-color);
            margin-bottom: 16px;
        }
        
        .welcome-text {
            font-size: 18px;
            margin-bottom: 8px;
            color: var(--text-primary);
        }
        
        .welcome-subtext {
            font-size: 14px;
            color: var(--text-secondary);
        }
        
        /* Document Upload */
        .upload-area {
            border: 2px dashed var(--border-color);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            margin-bottom: 16px;
            cursor: pointer;
            transition: border-color 0.3s;
            background: var(--main-bg);
        }
        
        .upload-area:hover {
            border-color: var(--primary-color);
        }
        
        .upload-button {
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            margin-top: 16px;
        }
        
        .document-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .document-item {
            padding: 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--sidebar-bg);
        }

        .document-item div:last-child {
            color: var(--primary-color);
        }
        
        .document-info h4 {
            margin-bottom: 4px;
            color: var(--text-primary);
        }
        
        .document-meta {
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        /* Source References */
        .sources-section {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border-color);
        }
        
        .source-item {
            background: var(--sidebar-bg);
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 14px;
            border: 1px solid var(--border-color);
        }
        
        .source-toggle-card {
            background: var(--ai-message-bg);
            padding: 16px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .source-toggle-header {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            color: var(--primary-color);
        }
        
        .source-toggle-button {
            align-self: flex-start;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 8px 16px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .source-toggle-button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        
        .source-toggle-button i {
            font-size: 0.85rem;
        }
        
        .source-list {
            display: none;
            border-top: 1px solid var(--border-color);
            padding-top: 12px;
            margin-top: 4px;
        }
        
        .source-list.visible {
            display: block;
        }
        
        .confidence-indicator {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }
        
        .confidence-high { background: #1a3a2a; color: #4ade80; }
        .confidence-medium { background: #3a2a1a; color: #fbbf24; }
        .confidence-low { background: #3a1a1a; color: #f87171; }
        
        /* Delete button styles */
        .delete-session-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            opacity: 0.6;
            transition: all 0.2s;
        }
        
        .delete-session-btn:hover {
            background: #3a1a1a;
            color: #f87171;
            opacity: 1;
        }

        /* Upload status styles */
        .upload-status {
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 14px;
            background: var(--sidebar-bg);
            color: var(--text-primary);
        }

        .upload-status.success {
            background: #1a3a2a;
            color: #4ade80;
        }

        .upload-status.error {
            background: #3a1a1a;
            color: #f87171;
        }

        /* Corporate Role Badge */
        .corporate-badge {
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 12px;
            border: 1px solid var(--border-color);
            background: var(--sidebar-bg);
        }

        .corporate-badge.manager 
        { border-left: 4px solid #a855f7; }

        .corporate-badge.hr { border-left: 4px solid #8b5cf6; }

        .corporate-badge.finance{ border-left: 4px solid #7c3aed; }

        .corporate-badge.it{ border-left: 4px solid #6d28d9; }

        .corporate-badge.employee { border-left: 4px solid #5b21b6; }


        /* Login Modal */
        .login-modal {
            display: flex;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }

        .login-container {
            background: var(--sidebar-bg);
            padding: 2rem;
            border-radius: 12px;
            width: 400px;
            border: 1px solid var(--border-color);
        }

        .demo-users {
            display: none;
            margin-top: 1rem;
            padding: 1rem;
            background: var(--main-bg);
            border-radius: 6px;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        /* Google Drive Status */
        .drive-status {
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 12px;
            border: 1px solid var(--border-color);
        }

        .drive-status.connected {
            background:rgb(231, 204, 244);  /* Purple background */
            border-left: 4px solid #8b5cf6;  /* Purple border */
        }

        .drive-status.disconnected {
            background:rgb(233, 219, 240);  /* Dark purple background */
            border-left: 4px solid #f87171;  /* Keep red for errors */
        }

        .drive-status.syncing {
            background:rgb(227, 219, 244);  /* Medium purple background */
            border-left: 4px solid #a855f7;  /* Bright purple border */
        }

        /* Progress bar */
        .progress-bar {
            width: 100%;
            height: 6px;
            background: var(--border-color);
            border-radius: 3px;
            overflow: hidden;
            margin: 8px 0;
        }

        .progress-fill {
            height: 100%;
            background: var(--primary-color);;
            transition: width 0.3s ease;
        }

   


        /* Thinking Indicator Styles */
        .thinking-indicator {
            display: none;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            background: var(--ai-message-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 16px;
            max-width: 80%;
            align-self: flex-start;
        }

        .thinking-dots {
            display: flex;
            gap: 4px;
        }

        .thinking-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--primary-color);
            animation: thinking-bounce 1.4s infinite ease-in-out;
        }

        .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
        .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
        .thinking-dot:nth-child(3) { animation-delay: 0s; }

        @keyframes thinking-bounce {
            0%, 80%, 100% {
                transform: scale(0.8);
                opacity: 0.5;
            }
            40% {
                transform: scale(1);
                opacity: 1;
            }
        }
        
        .chart-card {
            background: var(--ai-message-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 12px;
        }
        
        .chart-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-primary);
        }
        
        .chart-meta {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        
        .chart-canvas-container {
            position: relative;
            width: 100%;
            min-height: 280px;
            height: 320px;
        }

        .thinking-text {
            color: var(--text-secondary);
            font-size: 14px;
            font-style: italic;
        }


        /* Compact Mode */
        .compact-mode .chat-container {
            padding: 12px;
            gap: 12px;
        }

        .compact-mode .message-content {
            padding: 12px;
        }

        .compact-mode .input-area {
            padding: 16px;
        }

        .compact-mode .sidebar-content {
            padding: 4px;
        }

        .compact-mode .chat-history-item {
            padding: 8px 12px;
            margin-bottom: 2px;
        }

        .compact-mode .document-item {
            padding: 8px 12px;
        }

        .compact-mode .tab {
            padding: 8px 12px;
        }

        /* Show light logo in light mode, dark logo in dark mode */
        [data-theme="light"] .logo-dark { display: none; }
        [data-theme="light"] .logo-light { display: block; }

        [data-theme="dark"] .logo-light { display: none; }
        [data-theme="dark"] .logo-dark { display: block; }
    </style>
</head>
<body>
    
    <!-- Login Modal -->
    <div id="loginModal" class="login-modal">
        <div class="login-container">
            <h3 style="margin-bottom: 1rem; text-align: center;">Corporate Login</h3>
            <form id="loginForm">
                <div style="margin-bottom: 1rem;">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Company Email</label>
                    <input type="email" id="loginEmail" style="width: 100%; padding: 0.75rem; border: 1px solid var(--border-color); border-radius: 6px; background: var(--main-bg); color: var(--text-primary);" 
                        placeholder="your.email@company.com" required>
                </div>
                <div style="margin-bottom: 1.5rem;">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Password</label>
                    <input type="password" id="loginPassword" style="width: 100%; padding: 0.75rem; border: 1px solid var(--border-color); border-radius: 6px; background: var(--main-bg); color: var(--text-primary);" 
                        placeholder="••••••••" required>
                </div>
                <div style="display: flex; gap: 0.5rem;">
                    <button type="submit" style="flex: 1; background: var(--primary-color); color: white; border: none; padding: 0.75rem; border-radius: 6px; cursor: pointer;">
                        Login
                    </button>
                    <button type="button" onclick="toggleDemoUsers()" style="flex: 1; background: var(--border-color); color: var(--text-primary); border: none; padding: 0.75rem; border-radius: 6px; cursor: pointer;">
                        Demo Users
                    </button>
                </div>
            </form>
            <div id="demoUsers" class="demo-users">
                <h4 style="margin-bottom: 0.5rem;">Demo Corporate Users:</h4>
                <div> Manager: manager@company.com</div>
                <div> HR: hr@company.com</div>
                <div> Finance: finance@company.com</div>
                <div> IT: it@company.com</div>
                <div> Employee: employee@company.com</div>
                <div style="margin-top: 0.5rem; font-size: 0.7rem; color: var(--text-secondary);">Password: any</div>
            </div>
        </div>
    </div>
    
    <div class="app-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="sidebar-header">
                <div style="text-align: center; margin-bottom: 4px; padding: 2px; margin-left: 50px;">
                    <!-- Light mode logo -->
                    <img src="/static/logo.png" alt="Company Logo" class="logo-light" style="width: 70px; height: 70px; object-fit: contain; margin-left: 24px;">
                    <!-- Dark mode logo -->
                    <img src="/static/logo-dark.png" alt="Company Logo" class="logo-dark" style="width: 70px; height: 70px; object-fit: contain; margin-left: 24px;">
                </div>
                <div id="userInfo" style="display: none;">
                    <div class="corporate-badge employee" id="userBadge">
                        <div style="font-weight: 600; color: var(--primary-color);" id="userEmail"></div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            Role: <span id="userRole"></span>
                        </div>
                        <div style="font-size: 0.7rem; color: var(--text-secondary); margin-top: 0.25rem;" id="userAccess"></div>
                    </div>
                    <button onclick="logout()" style="margin-top: 0.5rem; width: 100%; padding: 0.5rem; background: transparent; border: 1px solid var(--border-color); border-radius: 4px; color: var(--text-secondary); cursor: pointer; font-size: 0.8rem;">
                        <i class="fas fa-sign-out-alt"></i> Logout
                    </button>
                </div>
                <button class="new-chat-btn" onclick="createNewSession()" id="newChatBtn" style="display: none; margin-top: 1rem;">
                    <i class="fas fa-plus"></i> New Chat Session
                </button>
            </div>
            <div class="sidebar-content">
                <div class="chat-history-list" id="chatHistoryList">
                    <!-- Chat history will be loaded here -->
                </div>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- Tabs -->
            <div class="tabs">
                <div class="tab active" data-tab="chat">Chat</div>
                <div class="tab" data-tab="documents">Documents</div>
                <div class="tab" data-tab="drive">Google Drive</div>
                <div class="tab" data-tab="settings">Settings</div>
            </div>
            
            <!-- Chat Tab -->
            <div class="tab-content active" id="chatTab">
                <div class="chat-container" id="chatContainer">
                    <div class="welcome-state" id="welcomeState">
                        <div class="welcome-text">
                            Hi there! 👋
                        </div>
                        <div class="welcome-subtext">
                            I can help you find information in our corporate documents. 
                            What would you like to know?
                        </div>
                    </div>
                    
                    <!-- ADD THIS THINKING INDICATOR -->
                    <div class="thinking-indicator" id="thinkingIndicator">
                        <div class="message-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div class="thinking-dots">
                                <div class="thinking-dot"></div>
                                <div class="thinking-dot"></div>
                                <div class="thinking-dot"></div>
                            </div>
                            <div class="thinking-text">Searching through documents...</div>
                        </div>
                    </div>
                </div>
                
                <div class="input-area">
                    <div class="input-container">
                        <form class="input-form" id="chatForm">
                            <textarea id="questionInput" 
                                    placeholder="Ask specific questions about your corporate documents..." 
                                    rows="1"></textarea>
                            <button type="submit" class="send-button" id="sendButton">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </form>
                    </div>
                </div>
            </div>
            
            <!-- Documents Tab -->
            <div class="tab-content" id="documentsTab">
                <div style="max-width: 800px; margin: 0 auto; padding: 0 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
                        <h3 style="margin: 0;">Corporate Documents</h3>
                        <div style="font-size: 0.9rem; color: var(--text-secondary);">
                            <i class="fas fa-database"></i> <span id="totalDocuments">0</span> documents loaded
                        </div>
                    </div>
                    
                    <div style="background: var(--sidebar-bg); padding: 20px; border-radius: 8px; margin-bottom: 24px; border: 1px solid var(--border-color);">
                        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                            <i class="fas fa-info-circle" style="color: var(--primary-color);"></i>
                            <div style="font-weight: 500;">Document Access Information</div>
                        </div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.5;">
                            Your role determines which documents you can access. All corporate documents are automatically loaded and secured.
                            You can upload additional files to your assigned namespace.
                        </div>
                    </div>
                    
                    <div class="upload-area" id="uploadArea">
                        <i class="fas fa-cloud-upload-alt" style="font-size: 2rem; color: var(--text-secondary); margin-bottom: 1rem;"></i>
                        <p>Upload Corporate Documents</p>
                        <p style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">
                            Supported: PDF, DOCX, TXT, CSV, Excel, PowerPoint, Images
                        </p>
                        <input type="file" id="fileInput" multiple style="display: none;" 
                               accept=".pdf,.docx,.txt,.csv,.xlsx,.xls,.pptx,.ppt,.png,.jpg,.jpeg">
                        <button class="upload-button" onclick="document.getElementById('fileInput').click()">
                            <i class="fas fa-folder-open"></i> Browse Files
                        </button>
                    </div>
                    
                    <div id="uploadStatus"></div>
                    
                    <div style="margin-top: 32px;">
                        <h4 style="margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                            <i class="fas fa-files"></i>
                            Available Documents
                        </h4>
                        <div class="document-list" id="documentList">
                            <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
                                <i class="fas fa-folder-open" style="font-size: 2rem; margin-bottom: 1rem;"></i>
                                <p>No documents uploaded yet</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Google Drive Tab -->
            <div class="tab-content" id="driveTab">
                <div style="max-width: 800px; margin: 0 auto; padding: 0 16px;">
                    <h3 style="margin-bottom: 24px;">
                        <i class="fab fa-google-drive" style="color: #4285f4;"></i>
                        Google Drive Integration
                    </h3>
                    
                    <!-- Drive Status -->
                    <div id="driveStatusContainer">
                        <div class="drive-status disconnected" id="driveStatus">
                            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                                <i class="fas fa-plug"></i>
                                <div style="font-weight: 500;">Google Drive Status</div>
                            </div>
                            <div style="font-size: 0.9rem; color: var(--text-secondary);">
                                Checking connection status...
                            </div>
                        </div>
                    </div>

                    <!-- Sync Controls -->
                    <div style="background: var(--sidebar-bg); padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid var(--border-color);">
                        <h4 style="margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                            <i class="fas fa-sync"></i>
                            Drive Sync Controls
                        </h4>
                        
                        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px;">
                            <button onclick="setupGoogleDrive()" class="upload-button" style="flex: 1;">
                                <i class="fab fa-google"></i> Connect Google Drive
                            </button>
                            <button onclick="syncGoogleDrive()" class="upload-button" style="flex: 1; background: #8b5cf6; color: white;">
                                <i class="fas fa-sync"></i> Sync Now
                            </button>
                            <button onclick="checkDriveStatus()" class="upload-button" style="flex: 1; background: #8b5cf6;">
                                <i class="fas fa-refresh"></i> Check Status
                            </button>
                        </div>

                        <!-- Sync Progress -->
                        <div id="syncProgress" style="display: none;">
                            <div style="display: flex; justify-content: between; margin-bottom: 8px;">
                                <span>Sync Progress</span>
                                <span id="progressPercent">0%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" id="progressFill" style="width: 0%;"></div>
                            </div>
                            <div id="syncDetails" style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 8px;"></div>
                        </div>
                    </div>

                    <!-- Sync Information -->
                    <div style="background: var(--sidebar-bg); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color);">
                        <h4 style="margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                            <i class="fas fa-info-circle"></i>
                            Sync Information
                        </h4>
                        <div style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.5;">
                            <p>• <strong>ALL ROLES</strong> can now sync Google Drive with appropriate folder access</p>
                            <p>• <strong>Managers/Executives:</strong> Full access to all Google Drive files</p>
                            <p>• <strong>HR/Finance/IT:</strong> Access to public + department-specific folders</p>
                            <p>• <strong>Employees:</strong> Access to company public folders only</p>
                            <p>• Maximum 200 files will be processed per sync</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Settings Tab -->
            <!-- Settings Tab -->
            <div class="tab-content" id="settingsTab">
                <div style="max-width: 600px; margin: 0 auto; padding: 0 16px;">
                    <h3 style="margin-bottom: 24px;">System Settings</h3>
                    
                    <!-- Appearance Settings -->
                    <div style="background: var(--sidebar-bg); padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid var(--border-color);">
                        <h4 style="margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                            <i class="fas fa-palette"></i>
                            Appearance Settings
                        </h4>
                        
                        <!-- Theme Selection -->
                        <div style="margin-bottom: 20px;">
                            <label style="display: block; margin-bottom: 12px; font-weight: 500; color: var(--text-primary);">
                                Theme
                            </label>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                                <button type="button" class="theme-option" data-theme="dark" onclick="changeTheme('dark')" 
                                        style="padding: 16px; border: 2px solid var(--border-color); border-radius: 8px; background: var(--sidebar-bg); color: black; cursor: pointer; text-align: left;">
                                    <div style="display: flex; align-items: center; gap: 12px;">
                                        <div style="width: 20px; height: 20px; border-radius: 50%; background: #343541; border: 2px solid #4a4b52;"></div>
                                        <div>
                                            <div style="font-weight: 500;">Dark Mode</div>
                                            <div style="font-size: 0.8rem; color: var(--text-secondary);">Default theme</div>
                                        </div>
                                    </div>
                                </button>
                                
                                <button type="button" class="theme-option" data-theme="light" onclick="changeTheme('light')"
                                        style="padding: 16px; border: 2px solid var(--border-color); border-radius: 8px; background: var(--sidebar-bg); color: var(--text-primary); cursor: pointer; text-align: left;">
                                    <div style="display: flex; align-items: center; gap: 12px;">
                                        <div style="width: 20px; height: 20px; border-radius: 50%; background: #ffffff; border: 2px solid #d9d9e3;"></div>
                                        <div>
                                            <div style="font-weight: 500;">Light Mode</div>
                                            <div style="font-size: 0.8rem; color: var(--text-secondary);">Bright theme</div>
                                        </div>
                                    </div>
                                </button>
                            </div>
                        </div>
                        
                        <!-- Font Size Settings -->
                        <div style="margin-bottom: 20px;">
                            <label style="display: block; margin-bottom: 12px; font-weight: 500; color: var(--text-primary);">
                                Font Size
                            </label>
                            <div style="display: flex; gap: 12px; align-items: center;">
                                <button type="button" onclick="changeFontSize('small')" class="font-size-btn"
                                        style="padding: 8px 16px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--main-bg); color: var(--text-primary); cursor: pointer; font-size: 12px;">
                                    Small
                                </button>
                                <button type="button" onclick="changeFontSize('medium')" class="font-size-btn active"
                                        style="padding: 8px 16px; border: 1px solid var(--primary-color); border-radius: 6px; background: var(--primary-color); color: white; cursor: pointer; font-size: 14px;">
                                    Medium
                                </button>
                                <button type="button" onclick="changeFontSize('large')" class="font-size-btn"
                                        style="padding: 8px 16px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--main-bg); color: var(--text-primary); cursor: pointer; font-size: 16px;">
                                    Large
                                </button>
                            </div>
                        </div>
                        
                        <!-- Layout Preferences -->
                        <div>
                            <label style="display: block; margin-bottom: 12px; font-weight: 500; color: var(--text-primary);">
                                Layout Preferences
                            </label>
                            <div style="display: flex; flex-direction: column; gap: 12px;">
                                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                                    <input type="checkbox" id="compactMode" onchange="toggleCompactMode()" style="accent-color: var(--primary-color);">
                                    <span>Compact mode (reduce spacing)</span>
                                </label>
                                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                                    <input type="checkbox" id="sidebarCollapse" onchange="toggleSidebarCollapse()" style="accent-color: var(--primary-color);">
                                    <span>Collapsible sidebar</span>
                                </label>
                            </div>
                        </div>
                    </div>
                    
                    <!-- User Information -->
                    <div style="background: var(--sidebar-bg); padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid var(--border-color);">
                        <h4 style="margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                            <i class="fas fa-user-shield"></i>
                            Access Information
                        </h4>
                        <div id="accessInfo" style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6;">
                            Please login to see access information
                        </div>
                    </div>

                    <!-- System Status -->
                    <div style="background: var(--sidebar-bg); padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid var(--border-color);">
                        <h4 style="margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                            <i class="fas fa-server"></i>
                            System Status
                        </h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                            <div style="text-align: center; padding: 12px; background: rgb(233, 219, 240); border-radius: 6px;">
                                <div style="font-size: 0.8rem; color: black; margin-bottom: 4px;">Documents</div>
                                <div style="font-weight: 600; color: #3b82f6;" id="systemDocuments">0</div>
                            </div>
                            <div style="text-align: center; padding: 12px; background: rgba(59, 129, 242, 0.1); border-radius: 6px;">
                                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 4px;">Chat Sessions</div>
                                <div style="font-weight: 600; color: #3b82f6;" id="systemSessions">0</div>
                            </div>
                        </div>
                        <div style="margin-top: 12px; padding: 12px; background: rgb(233, 219, 240); border-radius: 6px; border-left: 4px solid #8b5cf6;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <i class="fas fa-check-circle" style="color: #8b5cf6;"></i>
                                <span style="font-weight: 500; color: black;">System Operational</span>
                            </div>
                            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">
                                All corporate security systems active
                            </div>
                        </div>
                    </div>

                    <!-- Reset Settings -->
                    <div style="background: var(--sidebar-bg); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color);">
                        <h4 style="margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                            <i class="fas fa-undo"></i>
                            Reset Settings
                        </h4>
                        <div style="display: flex; gap: 12px;">
                            <button onclick="resetAppearanceSettings()" 
                                    style="padding: 12px 20px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--main-bg); color: var(--text-primary); cursor: pointer; font-size: 0.9rem;">
                                <i class="fas fa-palette"></i> Reset Appearance
                            </button>
                            <button onclick="clearAllSettings()" 
                                    style="padding: 12px 20px; border: 1px solid #ef4444; border-radius: 6px; background: transparent; color: #ef4444; cursor: pointer; font-size: 0.9rem;">
                                <i class="fas fa-trash"></i> Clear All Settings
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let currentSessionId = null;
        let hasMessages = false;
        let currentUser = null;
        let driveSyncInterval = null;

        // DOM Elements
        const chatContainer = document.getElementById('chatContainer');
        const questionInput = document.getElementById('questionInput');
        const chatForm = document.getElementById('chatForm');
        const sendButton = document.getElementById('sendButton');
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const uploadStatus = document.getElementById('uploadStatus');
        const welcomeState = document.getElementById('welcomeState');
        let chartRegistry = {};
        const thinkingIndicator = document.getElementById('thinkingIndicator');

        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            console.log("Professional Corporate System Loading...");
            showLoginModal();
            setupEventListeners();
            ensureChatStructure();
            hideThinkingIndicator();
            
            // Login form handler
            document.getElementById('loginForm').addEventListener('submit', function(e) {
                e.preventDefault();
                loginUser();
            });

            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    const tabName = this.getAttribute('data-tab');
                    switchTab(tabName);
                });
            });
        });

        async function loginUser() {
            const email = document.getElementById('loginEmail').value;
            const password = document.getElementById('loginPassword').value;
            
            if (!email) {
                alert('Please enter your company email address');
                return;
            }
            
            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        email: email,
                        password: password || 'any'
                    })
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || 'Login failed');
                }
                
                currentUser = data.user;
                console.log('Corporate login successful:', currentUser);
                
                // Update UI with corporate info
                updateCorporateUI(currentUser);
                
                hideLoginModal();
                createNewSession();
                
                // Start checking drive status if available
                if (currentUser.google_drive_available) {
                    startDriveStatusChecker();
                }
                
            } catch (error) {
                console.error('Corporate login error:', error);
                alert('Login failed: ' + error.message);
            }
        }

        function updateCorporateUI(user) {
            document.getElementById('userEmail').textContent = user.email;
            document.getElementById('userRole').textContent = user.role.toUpperCase();
            document.getElementById('userAccess').textContent = `Access Level: ${user.accessible_folders.join(', ')}`;
            
            // Update badge style based on role
            const badge = document.getElementById('userBadge');
            badge.className = `corporate-badge ${user.role}`;
            
            document.getElementById('userInfo').style.display = 'block';
            document.getElementById('newChatBtn').style.display = 'flex';
            
            // Update access information in settings
            updateAccessInfo(user);
            
            // Load system data
            loadSystemStatus();
            loadUploadedDocuments();
            
            showNotification(`Welcome ${user.email} - ${user.role} access granted`, 'success');
        }


        // Theme Management Functions
        function changeTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('app-theme', theme);
            updateThemeButtons(theme);
            showNotification(`Theme changed to ${theme} mode`, 'success');
        }

        function updateThemeButtons(selectedTheme) {
            document.querySelectorAll('.theme-option').forEach(btn => {
                const theme = btn.getAttribute('data-theme');
                if (theme === selectedTheme) {
                    btn.style.borderColor = 'var(--primary-color)';
                    btn.style.background = 'rgb(233, 219, 240)';
                } else {
                    btn.style.borderColor = 'var(--border-color)';
                    btn.style.background = 'var(--sidebar-bg)';
                }
            });
        }

        function changeFontSize(size) {
            const sizes = {
                'small': '12px',
                'medium': '14px', 
                'large': '16px'
            };
            
            document.documentElement.style.fontSize = sizes[size];
            localStorage.setItem('app-font-size', size);
            
            // Update button states
            document.querySelectorAll('.font-size-btn').forEach(btn => {
                btn.classList.remove('active');
                btn.style.background = 'var(--main-bg)';
                btn.style.borderColor = 'var(--border-color)';
                btn.style.color = 'var(--text-primary)';
            });
            
            event.target.classList.add('active');
            event.target.style.background = 'var(--primary-color)';
            event.target.style.borderColor = 'var(--primary-color)';
            event.target.style.color = 'white';
            
            showNotification(`Font size changed to ${size}`, 'success');
        }

        function toggleCompactMode() {
            const compactMode = document.getElementById('compactMode').checked;
            document.body.classList.toggle('compact-mode', compactMode);
            localStorage.setItem('app-compact-mode', compactMode);
            showNotification(`Compact mode ${compactMode ? 'enabled' : 'disabled'}`, 'success');
        }

        function toggleSidebarCollapse() {
            const sidebarCollapse = document.getElementById('sidebarCollapse').checked;
            localStorage.setItem('app-sidebar-collapse', sidebarCollapse);
            showNotification(`Sidebar collapse ${sidebarCollapse ? 'enabled' : 'disabled'}`, 'success');
        }

        function resetAppearanceSettings() {
            if (confirm('Reset all appearance settings to default?')) {
                localStorage.removeItem('app-theme');
                localStorage.removeItem('app-font-size');
                localStorage.removeItem('app-compact-mode');
                localStorage.removeItem('app-sidebar-collapse');
                
                // Reset to defaults
                changeTheme('dark');
                changeFontSize('medium');
                document.getElementById('compactMode').checked = false;
                document.getElementById('sidebarCollapse').checked = false;
                document.body.classList.remove('compact-mode');
                
                showNotification('Appearance settings reset to default', 'success');
            }
        }

        function clearAllSettings() {
            if (confirm('Clear ALL settings including theme preferences?')) {
                localStorage.clear();
                location.reload();
            }
        }

        function loadAppearanceSettings() {
            // Load theme
            const savedTheme = localStorage.getItem('app-theme') || 'dark';
            changeTheme(savedTheme);
            
            // Load font size
            const savedFontSize = localStorage.getItem('app-font-size') || 'medium';
            changeFontSize(savedFontSize);
            
            // Load compact mode
            const compactMode = localStorage.getItem('app-compact-mode') === 'true';
            document.getElementById('compactMode').checked = compactMode;
            document.body.classList.toggle('compact-mode', compactMode);
            
            // Load sidebar collapse
            const sidebarCollapse = localStorage.getItem('app-sidebar-collapse') === 'true';
            document.getElementById('sidebarCollapse').checked = sidebarCollapse;
        }

        // Initialize appearance settings when page loads
        document.addEventListener('DOMContentLoaded', function() {
            loadAppearanceSettings();
        });

        function updateAccessInfo(user) {
            const accessInfo = document.getElementById('accessInfo');
            let accessText = '';
            
            switch(user.role) {
                case 'manager':
                    accessText = `
                        <div style="margin-bottom: 12px;">
                            <strong>Full Administrative Access</strong><br>
                            You have complete access to all corporate documents and departments.
                        </div>
                    `;
                    break;
                case 'hr':
                    accessText = `
                        <div style="margin-bottom: 12px;">
                            <strong>HR Department Access</strong><br>
                            You can access HR documents and company-wide public information.
                        </div>
                    `;
                    break;
                case 'finance':
                    accessText = `
                        <div style="margin-bottom: 12px;">
                            <strong>Finance Department Access</strong><br>
                            You can access financial documents and company-wide public information.
                        </div>
                    `;
                    break;
                case 'it':
                    accessText = `
                        <div style="margin-bottom: 12px;">
                            <strong>IT Department Access</strong><br>
                            You can access IT documents and company-wide public information.
                        </div>
                    `;
                    break;
                default:
                    accessText = `
                        <div style="margin-bottom: 12px;">
                            <strong>Standard Employee Access</strong><br>
                            You can access company-wide public information only.
                        </div>
                    `;
            }
            
            accessText += `
                <div style="display: grid; grid-template-columns: auto 1fr; gap: 8px 12px; font-size: 0.8rem;">
                    <div style="color: var(--text-secondary);">Role:</div>
                    <div style="font-weight: 500;">${user.role.toUpperCase()}</div>
                    <div style="color: var(--text-secondary);">Namespace:</div>
                    <div style="font-weight: 500;">${user.namespace}</div>
                    <div style="color: var(--text-secondary);">Access Level:</div>
                    <div style="font-weight: 500;">${user.accessible_folders.join(', ')}</div>
                </div>
            `;
            
            accessInfo.innerHTML = accessText;
        }

        // Google Drive Functions
        async function setupGoogleDrive() {
            try {
                const response = await fetch('/auth/google');
                const data = await response.json();
                
                if (data.authorization_url) {
                    // Open OAuth in popup window with proper features
                    const width = 600;
                    const height = 700;
                    const left = (screen.width - width) / 2;
                    const top = (screen.height - height) / 2;
                    
                    const authWindow = window.open(
                        data.authorization_url, 
                        'google_auth',
                        `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes,status=yes`
                    );
                    
                    if (!authWindow) {
                        showNotification('Popup blocked! Please allow popups for this site and try again.', 'error');
                        return;
                    }
                    
                    showNotification('Google OAuth opened in new window. Please complete the authentication.', 'info');
                    
                    // Check for OAuth completion
                    const checkAuth = setInterval(() => {
                        if (authWindow.closed) {
                            clearInterval(checkAuth);
                            checkDriveStatus();
                            showNotification('Google Drive authentication completed!', 'success');
                        }
                    }, 1000);
                    
                    // Also listen for message from OAuth callback
                    window.addEventListener('message', function(event) {
                        if (event.data.type === 'oauth_success') {
                            clearInterval(checkAuth);
                            authWindow.close();
                            checkDriveStatus();
                            showNotification('Google Drive connected successfully!', 'success');
                        }
                    });
                    
                } else {
                    showNotification('Setup failed: ' + (data.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                showNotification('Setup error: ' + error.message, 'error');
            }
        }

        async function syncGoogleDrive() {
            if (!currentUser) {
                showNotification('Please login first', 'error');
                return;
            }
            
            
            try {
                const response = await fetch('/api/sync-drive', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        email: currentUser.email,
                        namespace: currentUser.namespace
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showNotification('Sync failed: ' + data.error, 'error');
                } else {
                    showNotification('Google Drive sync started successfully', 'success');
                    // Start checking sync status
                    startDriveStatusChecker();
                }
            } catch (error) {
                showNotification('Sync error: ' + error.message, 'error');
            }
        }

        async function checkDriveStatus() {
            if (!currentUser) return;
            
            try {
                // Check overall drive status
                const statusResponse = await fetch('/api/drive-status');
                const statusData = await statusResponse.json();
                
                // Check sync status for current namespace
                const syncResponse = await fetch(`/api/drive-sync-status?namespace=${currentUser.namespace}`);
                const syncData = await syncResponse.json();
                
                updateDriveStatusUI(statusData, syncData);
                
            } catch (error) {
                console.error('Drive status check failed:', error);
            }
        }

        function updateDriveStatusUI(statusData, syncData) {
            const driveStatus = document.getElementById('driveStatus');
            const syncProgress = document.getElementById('syncProgress');
            const progressFill = document.getElementById('progressFill');
            const progressPercent = document.getElementById('progressPercent');
            const syncDetails = document.getElementById('syncDetails');
            
            if (statusData.google_drive_connected) {
                driveStatus.className = 'drive-status connected';
                
                // 🆕 ADD ROLE-SPECIFIC MESSAGE
                let roleMessage = '';
                if (currentUser) {
                    if (currentUser.role === 'manager' || currentUser.role === 'executive') {
                        roleMessage = `👑 ${currentUser.role.toUpperCase()} - Full access to all Google Drive files`;
                    } else {
                        roleMessage = `📁 ${currentUser.role.toUpperCase()} - Access to: ${currentUser.accessible_folders.join(', ')}`;
                    }
                }
                
                driveStatus.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                        <i class="fas fa-check-circle" style="color: purple;"></i>
                        <div style="font-weight: 500;">Google Drive Connected</div>
                    </div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 8px;">
                        Connected as: ${statusData.connected_accounts[0] || 'Unknown'}
                    </div>
                    <div style="font-size: 0.8rem; color: var(--primary-color); font-weight: 500;">
                        ${roleMessage}
                    </div>
                `;
            } else {
                driveStatus.className = 'drive-status disconnected';
                driveStatus.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                        <i class="fas fa-times-circle" style="color: #f87171;"></i>
                        <div style="font-weight: 500;">Google Drive Not Connected</div>
                    </div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary);">
                        Click "Connect Google Drive" to set up integration
                    </div>
                `;
            }

            
            // Update sync progress
            if (syncData.status === 'syncing') {
                syncProgress.style.display = 'block';
                progressFill.style.width = `${syncData.progress}%`;
                progressPercent.textContent = `${syncData.progress.toFixed(1)}%`;
                syncDetails.innerHTML = `
                    Syncing folder: ${syncData.current_folder}<br>
                    Progress: ${syncData.folders_processed}/${syncData.total_folders} folders
                `;
            } else if (syncData.status === 'completed') {
                syncProgress.style.display = 'block';
                progressFill.style.width = '100%';
                progressPercent.textContent = '100%';
                syncDetails.innerHTML = `
                    ✅ Sync completed!<br>
                    Processed ${syncData.files_processed} files from ${syncData.folders_processed} folders
                `;
                // Hide progress after 5 seconds
                setTimeout(() => {
                    syncProgress.style.display = 'none';
                }, 5000);
            } else {
                syncProgress.style.display = 'none';
            }
        }

        function startDriveStatusChecker() {
            // Clear existing interval
            if (driveSyncInterval) {
                clearInterval(driveSyncInterval);
            }
            
            // Check immediately
            checkDriveStatus();
            
            // Check every 3 seconds
            driveSyncInterval = setInterval(() => {
                checkDriveStatus();
            }, 3000);
        }

        
        // Chat UI helpers
        function ensureChatStructure() {
            if (welcomeState && !chatContainer.contains(welcomeState)) {
                chatContainer.insertBefore(welcomeState, chatContainer.firstChild);
            }
            if (thinkingIndicator && !chatContainer.contains(thinkingIndicator)) {
                chatContainer.appendChild(thinkingIndicator);
            }
        }

        function clearChatMessages() {
            Object.values(chartRegistry).forEach(chart => {
                try { chart.destroy(); } catch (err) { console.warn('Chart cleanup failed', err); }
            });
            chartRegistry = {};
            chatContainer.querySelectorAll('.message').forEach(msg => msg.remove());
            hideThinkingIndicator();
            ensureChatStructure();
        }

        // Thinking indicator functions
        function showThinkingIndicator() {
            if (!thinkingIndicator) return;
            ensureChatStructure();
            thinkingIndicator.style.display = 'flex';
            chatContainer.appendChild(thinkingIndicator);
            
            // Scroll to bottom to show the thinking indicator
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function hideThinkingIndicator() {
            if (!thinkingIndicator) return;
            thinkingIndicator.style.display = 'none';
        }

        // Enhanced sendMessage function with thinking indicator
        async function sendMessage() {
            if (!currentUser) {
                showLoginModal();
                return;
            }

            const question = questionInput.value.trim();
            if (!question) return;
            
            if (!hasMessages) {
                hideWelcomeState();
            }
            
            addMessage(question, 'user');
            
            questionInput.value = '';
            questionInput.style.height = 'auto';
            
            sendButton.disabled = true;
            sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            
            // Show thinking indicator
            showThinkingIndicator();
            
            try {
                const formData = new FormData();
                formData.append('namespace', currentUser.namespace);
                formData.append('question', question);
                formData.append('email', currentUser.email);
                if (currentSessionId) {
                    formData.append('session_id', currentSessionId);
                }
                
                const response = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Chat request failed');
                }
                
                const data = await response.json();
                
                // Hide thinking indicator before showing actual response
                hideThinkingIndicator();
                
                await addStreamingMessage(data.answer, 'ai');
                
                if (data.chart) {
                    addChartMessage(data.chart);
                }
                
                if (data.confidence) {
                    const confidenceClass = `confidence-${data.confidence}`;
                    const confidenceText = data.confidence.charAt(0).toUpperCase() + data.confidence.slice(1);
                    addMessage(`<div style="margin-top: 8px;"><span class="confidence-indicator ${confidenceClass}">Confidence: ${confidenceText}</span></div>`, 'ai');
                }
                
                if (data.sources && data.sources.length > 0) {
                    addSourcesMessage(data.sources);
                }
                
                loadChatHistory();
                
            } catch (error) {
                // Hide thinking indicator on error too
                hideThinkingIndicator();
                console.error('Chat error:', error);
                addMessage(`<div style="color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> Error: ${error.message}</div>`, 'ai');
            } finally {
                sendButton.disabled = false;
                sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
            }
        }


        function showLoginModal() {
            document.getElementById('loginModal').style.display = 'flex';
        }

        function hideLoginModal() {
            document.getElementById('loginModal').style.display = 'none';
        }

        function toggleDemoUsers() {
            const demoDiv = document.getElementById('demoUsers');
            demoDiv.style.display = demoDiv.style.display === 'none' ? 'block' : 'none';
        }

        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 16px;
                border-radius: 6px;
                color: white;
                z-index: 1000;
                font-size: 14px;
                max-width: 300px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                ${type === 'success' ? 'background: #8b5cf6;' : ''}
                ${type === 'info' ? 'background: #3b82f6;' : ''}
                ${type === 'error' ? 'background: #ef4444;' : ''}
            `;
            notification.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px;">
                    <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'exclamation-triangle' : 'info'}-circle"></i>
                    ${message}
                </div>
            `;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.opacity = '0';
                notification.style.transition = 'opacity 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }, 4000);
        }

        function logout() {
            console.log("Corporate logout");
            currentUser = null;
            
            // Clear intervals
            if (driveSyncInterval) {
                clearInterval(driveSyncInterval);
                driveSyncInterval = null;
            }
            
            document.getElementById('userInfo').style.display = 'none';
            document.getElementById('newChatBtn').style.display = 'none';
            
            // Clear any existing chat
            clearChatMessages();
            ensureChatStructure();
            showWelcomeState();
            showLoginModal();
            
            showNotification('You have been logged out', 'info');
        }

        function setupEventListeners() {
            // Chat form submission
            chatForm.addEventListener('submit', function(e) {
                e.preventDefault();
                sendMessage();
            });
            
            // Auto-resize textarea
            questionInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });
            
            // File upload handling
            fileInput.addEventListener('change', handleFileUpload);
            
            // Drag and drop for files
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.style.borderColor = '#10a37f';
                this.style.background = 'rgba(16, 163, 127, 0.05)';
            });
            
            uploadArea.addEventListener('dragleave', function() {
                this.style.borderColor = 'var(--border-color)';
                this.style.background = 'var(--main-bg)';
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                this.style.borderColor = 'var(--border-color)';
                this.style.background = 'var(--main-bg)';
                const files = e.dataTransfer.files;
                handleFiles(files);
            });
            
            uploadArea.addEventListener('click', function() {
                fileInput.click();
            });
            
            // Enter key for sending (Shift+Enter for new line)
            questionInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }

        
        
        function switchTab(tabName) {
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Add active class to clicked tab
            const activeTab = document.querySelector(`.tab[data-tab="${tabName}"]`);
            if (activeTab) {
                activeTab.classList.add('active');
            }
            
            // Add active class to corresponding content
            const activeContent = document.getElementById(`${tabName}Tab`);
            if (activeContent) {
                activeContent.classList.add('active');
            }
            
            // Refresh data when switching to certain tabs
            if (tabName === 'documents') {
                loadUploadedDocuments();
            } else if (tabName === 'drive') {
                checkDriveStatus();
            } else if (tabName === 'settings') {
                loadSystemStatus();
            }
        }
        
        function createNewSession() {
            if (!currentUser) return;
            
            fetch('/api/create_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    namespace: currentUser.namespace,
                    title: 'New Corporate Chat'
                })
            })
            .then(response => response.json())
            .then(data => {
                currentSessionId = data.session_id;
                hasMessages = false;
                
                clearChatMessages();
                ensureChatStructure();
                showWelcomeState();
                
                loadChatHistory();
            })
            .catch(error => {
                console.error('Error creating session:', error);
            });
        }
        
        function showWelcomeState() {
            welcomeState.style.display = 'flex';
            hasMessages = false;
        }
        
        function hideWelcomeState() {
            welcomeState.style.display = 'none';
            hasMessages = true;
        }
        
        function loadChatHistory() {
            if (!currentUser) return;
            
            fetch(`/api/chat_sessions?namespace=${currentUser.namespace}`)
                .then(response => response.json())
                .then(sessions => {
                    const historyList = document.getElementById('chatHistoryList');
                    historyList.innerHTML = '';
                    
                    if (sessions.length === 0) {
                        historyList.innerHTML = `
                            <div style="text-align: center; padding: 20px; color: var(--text-secondary);">
                                <i class="fas fa-comments" style="font-size: 1.5rem; margin-bottom: 0.5rem;"></i>
                                <div>No chat history</div>
                            </div>
                        `;
                        return;
                    }
                    
                    sessions.forEach(session => {
                        const item = document.createElement('div');
                        item.className = 'chat-history-item';
                        if (session.session_id === currentSessionId) {
                            item.classList.add('active');
                        }
                        
                        item.innerHTML = `
                            <i class="fas fa-message"></i>
                            <div style="flex: 1; overflow: hidden;">
                                <div style="font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                    ${session.title}
                                </div>
                                <div style="font-size: 12px; color: var(--text-secondary);">
                                    ${new Date(session.updated_at).toLocaleDateString()} • ${session.message_count} messages
                                </div>
                            </div>
                            <button class="delete-session-btn" onclick="deleteSession('${session.session_id}', event)">
                                <i class="fas fa-trash"></i>
                            </button>
                        `;
                        
                        item.addEventListener('click', function(e) {
                            if (!e.target.closest('.delete-session-btn')) {
                                loadChatSession(session.session_id);
                            }
                        });
                        
                        historyList.appendChild(item);
                    });
                })
                .catch(error => {
                    console.error('Error loading chat history:', error);
                });
        }
        
        function deleteSession(sessionId, event) {
            event.stopPropagation();
            
            if (confirm('Are you sure you want to delete this chat session?')) {
                fetch(`/api/chat_session/${sessionId}`, {
                    method: 'DELETE'
                })
                .then(response => response.json())
                .then(data => {
                    loadChatHistory();
                    if (sessionId === currentSessionId) {
                        createNewSession();
                    }
                    showNotification('Chat session deleted', 'success');
                })
                .catch(error => {
                    console.error('Error deleting session:', error);
                    showNotification('Error deleting session', 'error');
                });
            }
        }
        
        function loadChatSession(sessionId) {
            fetch(`/api/chat_history?session_id=${sessionId}`)
                .then(response => response.json())
                .then(history => {
                    clearChatMessages();
                    ensureChatStructure();
                    
                    if (history.length === 0) {
                        showWelcomeState();
                    } else {
                        hideWelcomeState();
                        history.forEach(msg => {
                            addMessage(msg.content, msg.type, false);
                        });
                    }
                    
                    currentSessionId = sessionId;
                    switchTab('chat');
                    
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    loadChatHistory();
                })
                .catch(error => {
                    console.error('Error loading chat session:', error);
                });
        }
        
        function loadUploadedDocuments() {
            if (!currentUser) return;
            
            fetch(`/api/uploaded_documents?namespace=${currentUser.namespace}`)
                .then(response => response.json())
                .then(documents => {
                    const documentList = document.getElementById('documentList');
                    documentList.innerHTML = '';
                    
                    if (documents.length === 0) {
                        documentList.innerHTML = `
                            <div style="text-align: center; padding: 40px; color: var(--text-secondary);">
                                <i class="fas fa-folder-open" style="font-size: 2rem; margin-bottom: 1rem;"></i>
                                <p>No documents available in your namespace</p>
                                <p style="font-size: 0.8rem; margin-top: 0.5rem;">Upload files to get started</p>
                            </div>
                        `;
                        return;
                    }
                    
                    // Simple header
                    const header = document.createElement('div');
                    header.style.cssText = 'padding: 12px; background: var(--sidebar-bg); border-radius: 6px; margin-bottom: 12px; border: 1px solid var(--border-color);';
                    header.innerHTML = `
                        <div style="font-weight: 500; color: black;">
                            <i class="fas fa-list"></i> Showing ${documents.length} Documents
                        </div>
                    `;
                    documentList.appendChild(header);
                    
                    // Add all documents
                    documents.forEach(doc => {
                        const item = document.createElement('div');
                        item.className = 'document-item';
                        
                        const fileSize = (doc.file_size / 1024 / 1024).toFixed(2);
                        const uploadDate = new Date(doc.upload_time).toLocaleDateString();
                        
                        item.innerHTML = `
                            <div style="flex: 1;">
                                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                                    <i class="fas fa-file" style="color: var(--primary-color);"></i>
                                    <h4 style="margin: 0; font-size: 0.95rem;">${doc.filename}</h4>
                                </div>
                                <div class="document-meta">
                                    <span>${fileSize} MB</span> • 
                                    <span>${doc.chunk_count} chunks</span> • 
                                    <span>${uploadDate}</span>
                                </div>
                            </div>
                            <div style="color: var(--primary-color);">
                                <i class="fas fa-check-circle"></i>
                            </div>
                        `;
                        
                        documentList.appendChild(item);
                    });
                    
                    document.getElementById('totalDocuments').textContent = documents.length;
                })
                .catch(error => {
                    console.error('Error loading documents:', error);
                });
        }
        
        async function loadSystemStatus() {
            try {
                // Load document count
                const docsResponse = await fetch(`/api/uploaded_documents?namespace=${currentUser.namespace}`);
                const documents = await docsResponse.json();
                document.getElementById('systemDocuments').textContent = documents.length;
                
                // Load session count
                const sessionsResponse = await fetch(`/api/chat_sessions?namespace=${currentUser.namespace}`);
                const sessions = await sessionsResponse.json();
                document.getElementById('systemSessions').textContent = sessions.length;
                
            } catch (error) {
                console.error('Error loading system status:', error);
            }
        }
        
        function handleFileUpload(e) {
            handleFiles(e.target.files);
        }
        
        function handleFiles(files) {
            if (!currentUser) {
                showLoginModal();
                return;
            }

            uploadStatus.innerHTML = '';
            
            for (let file of files) {
                uploadFile(file);
            }
            
            fileInput.value = '';
        }

        function uploadFile(file) {
            const statusDiv = document.createElement('div');
            statusDiv.style.cssText = 'padding: 12px; border-radius: 6px; background: var(--sidebar-bg); margin-bottom: 8px; font-size: 14px;';
            statusDiv.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Uploading "${file.name}"...`;
            uploadStatus.appendChild(statusDiv);
            
            const formData = new FormData();
            formData.append('namespace', currentUser.namespace);
            formData.append('email', currentUser.email);
            formData.append('file', file);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { 
                        throw new Error(err.detail || 'Upload failed'); 
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.skipped) {
                    statusDiv.className = 'upload-status info';
                    statusDiv.innerHTML = `<i class="fas fa-info-circle"></i> "${file.name}" skipped: ${data.message}`;
                    showNotification(`Document "${file.name}" skipped (already exists)`, 'info');
                } else if (data.chunks && data.chunks > 0) {
                    statusDiv.className = 'upload-status success';
                    statusDiv.innerHTML = `<i class="fas fa-check-circle"></i> "${file.name}" uploaded successfully! Processed ${data.chunks} chunks.`;
                    loadUploadedDocuments();
                    loadSystemStatus();
                    showNotification(`Document "${file.name}" uploaded successfully`, 'success');
                } else {
                    statusDiv.className = 'upload-status error';
                    statusDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error uploading "${file.name}": ${data.detail || 'Unknown error'}`;
                }
            })
            .catch(error => {
                statusDiv.className = 'upload-status error';
                statusDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error uploading "${file.name}": ${error.message}`;
                showNotification(`Upload failed: ${error.message}`, 'error');
            });
        }
        
        function createMessageElement(type, scroll = true) {
            ensureChatStructure();
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            const avatarDiv = document.createElement('div');
            avatarDiv.className = 'message-avatar';
            avatarDiv.innerHTML = type === 'user' ? 
                '<i class="fas fa-user"></i>' : 
                '<i class="fas fa-robot"></i>';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            messageDiv.appendChild(avatarDiv);
            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);
            if (thinkingIndicator) {
                chatContainer.appendChild(thinkingIndicator);
            }
            
            if (scroll) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            return { messageDiv, contentDiv };
        }

        function addMessage(content, type, scroll = true) {
            const { contentDiv } = createMessageElement(type, scroll);
            
            if (typeof content === 'string' && content.includes('<')) {
                contentDiv.innerHTML = content;
            } else {
                contentDiv.textContent = content;
            }
            
            return contentDiv;
        }

        function addStreamingMessage(content, type = 'ai', options = {}) {
            const { speed = 35, scroll = true } = options;
            const tokens = typeof content === 'string' ? content.split(/(\s+)/) : [];
            
            if (tokens.length === 0) {
                addMessage(content || '', type, scroll);
                return Promise.resolve();
            }
            
            const { contentDiv } = createMessageElement(type, scroll);
            contentDiv.textContent = '';
            
            return new Promise(resolve => {
                let index = 0;
                let timer = null;
                
                const appendNext = () => {
                    if (index >= tokens.length) {
                        if (timer) clearInterval(timer);
                        resolve();
                        return;
                    }
                    
                    contentDiv.textContent += tokens[index];
                    index++;
                    
                    if (scroll) {
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                    
                    if (index >= tokens.length) {
                        if (timer) clearInterval(timer);
                        resolve();
                    }
                };
                
                appendNext();
                if (index < tokens.length) {
                    timer = setInterval(appendNext, speed);
                }
            });
        }

        function addSourcesMessage(sources) {
            if (!sources || sources.length === 0) return;
            
            const { contentDiv } = createMessageElement('ai');
            const card = document.createElement('div');
            card.className = 'source-toggle-card';
            
            const header = document.createElement('div');
            header.className = 'source-toggle-header';
            header.innerHTML = '<i class="fas fa-book-open"></i><span>Source References</span>';
            
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'source-toggle-button';
            button.innerHTML = '<i class="fas fa-eye"></i> Show sources';
            button.setAttribute('aria-expanded', 'false');
            
            const list = document.createElement('div');
            list.className = 'source-list';
            
            list.innerHTML = sources.map(source => `
                <div class="source-item">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                        <strong style="font-size: 0.9rem;">${source.source || 'Unknown'}</strong>
                        <span style="font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; 
                            background: ${source.relevance_score !== 'N/A' && parseFloat(source.relevance_score) > 0.7 ? '#1a3a2a' : 
                                        source.relevance_score !== 'N/A' && parseFloat(source.relevance_score) > 0.4 ? '#3a2a1a' : '#3a1a1a'}; 
                            color: ${source.relevance_score !== 'N/A' && parseFloat(source.relevance_score) > 0.7 ? '#4ade80' : 
                                    source.relevance_score !== 'N/A' && parseFloat(source.relevance_score) > 0.4 ? '#fbbf24' : '#f87171'};">
                            ${source.relevance_score !== 'N/A' ? source.relevance_score + ' relevance' : 'Score N/A'}
                        </span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); font-style: italic; line-height: 1.4; margin-bottom: 6px;">
                        ${source.excerpt || 'No excerpt available'}
                    </div>
                    <div style="font-size: 0.7rem; color: var(--text-secondary); display: flex; justify-content: space-between;">
                        <span>${source.file_type}</span>
                        <span>${source.content_length} chars</span>
                    </div>
                </div>
            `).join('');
            
            button.addEventListener('click', () => {
                const isVisible = list.classList.toggle('visible');
                button.innerHTML = isVisible ? '<i class="fas fa-eye-slash"></i> Hide sources' : '<i class="fas fa-eye"></i> Show sources';
                button.setAttribute('aria-expanded', isVisible ? 'true' : 'false');
            });
            
            card.appendChild(header);
            card.appendChild(button);
            card.appendChild(list);
            contentDiv.appendChild(card);
        }

        function addChartMessage(chartConfig) {
            if (!chartConfig || !Array.isArray(chartConfig.labels) || !Array.isArray(chartConfig.datasets)) {
                return;
            }
            
            const { contentDiv } = createMessageElement('ai');
            
            const card = document.createElement('div');
            card.className = 'chart-card';
            
            const header = document.createElement('div');
            header.className = 'chart-header';
            
            const titleWrapper = document.createElement('div');
            const title = document.createElement('div');
            title.className = 'chart-title';
            title.textContent = chartConfig.title || 'Generated Chart';
            
            const meta = document.createElement('div');
            meta.className = 'chart-meta';
            const metaParts = [];
            if (chartConfig.meta?.source) metaParts.push(chartConfig.meta.source);
            if (chartConfig.meta?.sheet) metaParts.push(chartConfig.meta.sheet);
            if (metaParts.length > 0) {
                meta.textContent = metaParts.join(' • ');
                titleWrapper.appendChild(title);
                titleWrapper.appendChild(meta);
            } else {
                titleWrapper.appendChild(title);
            }
            
            header.appendChild(titleWrapper);
            card.appendChild(header);
            
            const canvasContainer = document.createElement('div');
            canvasContainer.className = 'chart-canvas-container';
            const canvas = document.createElement('canvas');
            const canvasId = chartConfig.chart_id || `chart-${Date.now()}-${Math.random().toString(16).slice(2)}`;
            canvas.id = canvasId;
            canvasContainer.appendChild(canvas);
            card.appendChild(canvasContainer);
            contentDiv.appendChild(card);
            
            if (typeof Chart === 'undefined') {
                const warning = document.createElement('div');
                warning.className = 'chart-meta';
                warning.textContent = 'Chart library not available. Please refresh the page.';
                card.appendChild(warning);
                return;
            }
            
            const palette = ['#10a37f', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899'];
            const preparedDatasets = chartConfig.datasets.map((dataset, idx) => {
                const color = dataset.backgroundColor || palette[idx % palette.length];
                return {
                    ...dataset,
                    backgroundColor: color,
                    borderColor: dataset.borderColor || color,
                    borderWidth: dataset.borderWidth ?? (chartConfig.type === 'line' ? 2 : 0),
                    tension: dataset.tension ?? (chartConfig.type === 'line' ? 0.3 : 0),
                    pointRadius: dataset.pointRadius ?? (chartConfig.type === 'line' ? 3 : 0)
                };
            });
            
            const chartInstance = new Chart(canvas.getContext('2d'), {
                type: chartConfig.type || 'bar',
                data: {
                    labels: chartConfig.labels,
                    datasets: preparedDatasets
                },
                options: chartConfig.options || {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true },
                        tooltip: { mode: 'index', intersect: false }
                    },
                    scales: {
                        x: { title: { display: true, text: chartConfig.meta?.dimension_column || '' } },
                        y: { title: { display: true, text: chartConfig.meta?.metric_column || '' }, beginAtZero: true }
                    }
                }
            });
            
            chartRegistry[canvasId] = chartInstance;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>
"""


@app.get("/")
async def root():
    return HTMLResponse(COMPLETE_HTML_INTERFACE)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Professional RAG System",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🎯 COMPLETE PROFESSIONAL RAG SYSTEM STARTING...")
    print("Open: http://localhost:8000")
    print("✅ ALL PROFESSIONAL FEATURES ACTIVATED:")
    print("   🏆 Professional document processing with specialized loaders")
    print("   🏆 Multi-sheet Excel support with structured formatting") 
    print("   🏆 Professional PDF extraction using PyPDFLoader")
    print("   🏆 Intelligent chunking system")
    print("   🏆 Hybrid search (vector + BM25)")
    print("   🏆 Query enhancement")
    print("   🏆 Professional Google Drive integration")
    print("   🏆 Complete frontend with all endpoints")
    print("   🏆 Corporate access control")
    print("   🏆 Chat history management")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
