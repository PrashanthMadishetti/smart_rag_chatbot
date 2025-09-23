from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from typing import List,Optional
from os import path
import hashlib
from datetime import datetime,timezone
from pathlib import Path
from bs4 import BeautifulSoup
import requests


#utilities

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _iso_from_mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()


def _normalize_text(s:str) -> str:
    '''
    Canonical text normalization used for hashing and storage:
    - remove zero-width/control characters
    - strip lines and drop empty ones
    - preserve paragraph boundaries via '\n'
    '''
    if not s:
         return ""
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    lines = [ln.strip() for ln in s.splitlines()]
    lines =[ln for ln in lines if ln]
    return "\n".join(lines)

def _sha256_hex(data:bytes)-> str:
    return hashlib.sha256(data).hexdigest()

def _doc_id_from_text(normalized_text:str) -> str:
    '''
    Creates a document id
    '''
    h_text = _sha256_hex(normalized_text.encode('utf-8'))
    raw = f"doc:v1|{h_text}".encode("utf-8")
    return _sha256_hex(raw)

def _page_id_from_text(doc_id:str, page_number:int, page_text_norm:str) -> str:
    '''
    Creates a per page stable id tied to doc_id + page_number + page content
    '''
    h_page = _sha256_hex(page_text_norm.encode('utf-8'))
    raw = f"page:v1|{doc_id}|{page_number}|{h_page}".encode("utf-8")
    return _sha256_hex(raw)

#----------------
# Web Helpers
#----------------

def _html_to_text_and_title(html: str) -> tuple[str, Optional[str]]:
    '''
    Retrieves the content of the web page, strips the html tags and return the normalized content
    '''
    soup = BeautifulSoup(html,"html.parser")

    for tag in soup(["script", "style","noscript"]):
        tag.decompose()
    
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    text = soup.get_text(" ",strip=True)
    return _normalize_text(text), title


    
#----------------
# PDF Loader
#----------------

def load_pdfs(paths:List[str])-> List[Document]:
    '''
    loads PDF and returns a list of langchain documents(contains page_content and metadata)
    '''
    docs: List[Document]= []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {p}")
       
        #Load and normalize all pages to compute the doc_id
        ''' Paragraph boundaries are preserved using a '\n' and page boundaries 
            are preserved using '\n\n'
        '''
        pages = PyPDFLoader(path).load_and_split()
        normalized_pages = [_normalize_text(page.page_content) for page in pages]
        full_text_norm = "\n\n".join(normalized_pages)
        doc_id = _doc_id_from_text(full_text_norm)


        for idx, page_norm in enumerate(normalized_pages,start =1):
            page_id = _page_id_from_text(doc_id=doc_id, page_number=idx, page_text_norm=page_norm)
            checksum = _sha256_hex(page_norm.encode("utf-8"))

            md = {
                "source": str(path.resolve()),
                "type":"pdf",
                "doc_id":doc_id,
                "timestamp":_now_iso(),
                "page":idx,
                "page_id":page_id,
                "title":path.name,
                "checksum":checksum,
                "id_version":"v1"
            }

            docs.append(Document(page_content=page_norm,metadata=md))
    return docs


#----------------
# Txt Loader
#----------------

def load_txts(paths:List[str]) -> List[Document]:
    '''
    loads txt documents and returns a list of langchain documents
    '''
    docs:List[Document] = []
    for p in paths:
        path = Path(p)
        if not path:
            raise FileNotFoundError(f"Missing file:{p}")
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = _normalize_text(raw)
        doc_id = _doc_id_from_text(text)
        checksum = _sha256_hex(text.encode("utf-8"))

        md = {
            "source":path,
            "type":"txt",
            "doc_id":doc_id,
            "timestamp":_now_iso(),
            "file_mtime": _iso_from_mtime(path),
            "title":path.name,
            "checksum":checksum,
            "id_version":"v1"
        }

        docs.append(Document(page_content = text,metadata = md))
    return docs



#----------------
# Web Loader
#----------------

def load_web(urls:List[str]) -> List[Document]:
    '''
    loads the web page and returns a list of Documents
    '''
    docs:List[Document] = []
    for url in urls:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        resolved_url = str(resp.url)
        text_norm, title = _html_to_text_and_title(resp.text)
        doc_id = _doc_id_from_text(text_norm)
        checksum = _sha256_hex(text_norm.encode("utf-8"))
        now = _now_iso()

        md = {
            "source":resolved_url,
            "type":"web",
            "url":url,
            "resolved_url":resolved_url,
            "doc_id":doc_id,
            "timestamp":now,
            "title":title,
            "checksum":checksum,
            "id_version":"v1"
        }

        docs.append(Document(page_content=text_norm,metadata = md))
    return docs

