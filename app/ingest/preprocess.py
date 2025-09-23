from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from typing import List,Tuple,Iterable
import hashlib
from dataclasses import dataclass

#ZW spaces/joind + BOM
_ZEROWIDTH = re.compile(r"[\u200B\u200C\u200D\u2060\ufeff]")
#ASCII control characters
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
#Collapses runs of spaces/tabs but not new lines; we'll normalize lines then rejoin with \n
_MULTI_SPACE = re.compile(r"[ \t]+")

def _collapse_spaces_keep_bullets(line:str) -> str:
    '''
    Collapse redundant spaces but keep markers
    '''
    bullet_prefix = None
    if line.startswith("- "):
        bullet_prefix = "- "
        core = line[2:]
    elif line.startswith("-"):
        bullet_prefix = "- "
        core = line[1:].lstrip()
    elif line.startswith("• "):
        bullet_prefix = "• "
        core = line[2:]
    elif line.startswith("•"):
        bullet_prefix = "• "
        core = line[1:].lstrip()
    else:
        core = line

    core = _MULTI_SPACE.sub(" ", core.strip())
    return (bullet_prefix + core) if bullet_prefix else core


def _normalize_text_keep_paragraphs(text:str) -> str:
    '''
    Canonical Text Normalization used for cleaning:
    - remove zero width and control characters
    - trim lines, collapse inner spaces/tabs
    - preserve paragraph boundaries(blank lines become single blank line)
    '''
    if not text:
        return ""
    
    #Strip Zero-width + control characters but keep \n and \t
    text = _ZEROWIDTH.sub(" ", text)
    text = _CONTROL.sub(" ",text)

    #Normalize Windows/Mac line endings to \n
    text = text.replace("\r\n","\n").replace("\r","\n")

    #Split into lines, trim, collapse spaces on each line
    lines = [ln.rstrip() for ln in text.split("\n")]

    norm_lines:List[str] =[]
    blank_pending = False
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            #Collapse multiple blank lines into a Single blank line
            blank_pending = True
            continue
        
        if blank_pending and norm_lines:
            norm_lines.append("") #Insert a single blank line between paragraphs
            blank_pending = False
        norm_lines.append(_collapse_spaces_keep_bullets(stripped))
    return "\n".join(norm_lines)




def clean(doc:Document) -> Document:
    '''
    Reserve a cleaned copy of the input document, preserving metadata 
    '''
    cleaned = _normalize_text_keep_paragraphs(doc.page_content or "")
    return Document(page_content=cleaned, metadata = dict(doc.metadata))




#-------------
# Chunking
#-------------

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

@dataclass(frozen=True)
class _ChunkingConfig:
    chunk_size:int = 800
    chunk_overlap:int = 120
    separators: Tuple[str, ...] =("\n\n", "\n", " ","")

def _reset_scope_key(md:dict) -> Tuple[str,int]:
    doc_id = md.get("doc_id","")
    if md.get("type") == "pdf" and "page" in md:
        return (md["doc_id"],md["page"])
    return (md["doc_id"],0)

def _split_text(text:str, cfg: _ChunkingConfig) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = cfg.chunk_size, chunk_overlap = cfg.chunk_overlap, separators = cfg.separators
    )
    return splitter.split_text(text)

def chunk(
        docs: Iterable[Document],
        chunk_size: int = 800,
        chunk_overlap:int = 120,
        separators: Tuple[str, ...] =("\n\n", "\n", " ","")
) -> List[Document]:
    '''
    Split input Documents into overlapping chunks with inherited metadata and 
    added fields: chunk_id, chunk_uid, chunk_checksum

    chunk_uid format: "{doc_id}:{page or 0}:{chunk_id}"
    '''
    cfg = _ChunkingConfig(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=separators)
    out:List[Document] = []


    #Track counters to reset scope
    counters: dict[(str,int),int]={}

    for doc in docs:
        base_md = dict(doc.metadata) if doc.metadata else {}
        scope = _reset_scope_key(base_md)
        counters.setdefault(scope,0)

        parts = _split_text(doc.page_content or "",cfg)
        for part in parts:
            chunk_id = counters[scope]
            counters[scope] = chunk_id + 1

            page_num = scope[1]
            doc_id = base_md.get("doc_id", "")
            chunk_uid = f"{doc_id}:{page_num}:{chunk_id}"
            checksum = _sha256_hex(part.encode("utf-8"))

            md = dict(base_md)
            md.update(
                {
                    "chunk_id":chunk_id,
                    "chunk_uid":chunk_uid,
                    "chunk_checksum":checksum
                }
            )

            out.append(Document(page_content=part,metadata = md))
    return out

        



