import os, uuid, shutil
from pathlib import Path
from typing import Iterable, Union, List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document  # or from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile


# Detect if it's a Streamlit UploadedFile (avoid strict import here)
def _is_uploaded_file(obj) -> bool:
    return hasattr(obj, "read") and hasattr(obj, "name")

def _save_uploaded_to_temp(uploaded_file) -> Path:
    # preserve original name but ensure uniqueness on disk
    safe_name = Path(uploaded_file.name).name
    tmp_name = f"{uuid.uuid4()}_{safe_name}"
    tmp_path = Path("tmp_uploads") / tmp_name
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    # Use .getvalue() to avoid pointer issues
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return tmp_path

def process_pdf(input_obj: Union[str, os.PathLike, "UploadedFile"]) -> List[Document]:
    """
    Accepts either:
      - str/path: path to an existing PDF on disk
      - Streamlit UploadedFile
    Returns: list[Document] with metadata['source'] (display name) and metadata['path'] (actual path)
    """
    # Resolve source path & display name
    if _is_uploaded_file(input_obj):
        display_name = Path(input_obj.name).name
        pdf_path = _save_uploaded_to_temp(input_obj)
    else:
        pdf_path = Path(input_obj)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        display_name = pdf_path.name

    # Load pages
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    # Normalize metadata
    for doc in pages:
        doc.metadata = dict(doc.metadata or {})
        doc.metadata.setdefault("path", str(pdf_path))      # actual on-disk path used
        doc.metadata.setdefault("source", display_name)     # nice display name for UI

    return pages

def process_many(inputs: Iterable[Union[str, os.PathLike, "UploadedFile"]]) -> List[Document]:
    """Process a mix of file paths and UploadedFiles, return a single list of Documents."""
    all_docs: List[Document] = []
    for item in inputs:
        all_docs.extend(process_pdf(item))
    return all_docs
