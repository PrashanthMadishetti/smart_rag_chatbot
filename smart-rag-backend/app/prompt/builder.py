from typing import Dict,Any,List,Optional,Tuple,Iterable
from dataclasses import dataclass

from langchain_core.documents import Document

@dataclass(frozen=True)
class PromptBuilderConfig:
    max_context_chars:int = 4000
    max_history_chars:int = 2000

    headroom_chars:int = 600

    citation_style:str = "inline"

    refusal_line:str = (
        "If the answer is not grounded in the provided context, say"
        "\"I don't know based on the knowledge base.\"Do not hallucinate."
    )
    system_rules:str = (
        "You are a helpful RAG assistant. Answer **only** using the provided Context. "
        "Do not use outside knowledge. Always include inline citations next to claims "
        "in the form [source:chunk_id]. Be concise, factual, and cite each distinct fact."
    )

def _format_context_inline(chunks:List[Document]) -> str:
    lines:List[str] = ["Context:"]
    for i,d in enumerate(chunks,1):
        src = str(d.metadata.get("source","unknown"))
        cid = str(d.metadata.get("chunk_id",f"{i}"))
        body = d.page_content or ""
        lines.append(f"- {src} #{cid} {body}")
    return "\n".join(lines)

def _size(s:str)->int:
    return len(s or "")

def _reduce_context_to_budget(chunks:List[Document], budget:int) -> List[Document]:
    if not chunks:
        return []
    kept:List[Document] = []
    for d in chunks:
        trial = _format_context_inline(kept+[d])
        if _size(trial)< budget:
            kept.append(d)
        else:
            break
    return kept

def _render_history(history:Iterable[Tuple[str,str]]) -> str:
    lines:List[str] = ["History:"]
    for u,a in history:
        if u:
            lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistant: {a}")
    return "\n".join(lines)



def _reduce_history_to_budget(history:List[Tuple[str,str]],budget:int) -> List[Tuple[str,str]]:
    if not history:
        return []
    start = 0
    while start < len(history):
        rendered = _render_history(history[start:])
        if _size(rendered) <= budget:
            return history[start:]
        start +=1
    return []

                                                                    
def _collect_sources(chunks:List[Document]) -> List[Dict[str,str]]:
    out:List[Dict[str,str]] = []
    for d in chunks:
        out.append({
            "source":str(d.metadata.get("source","unknown")),
            "chunk_id":str(d.metadata.get("chunk_id","")),
        })
    return out

def build_prompt(
    *,
    query:str,
    chunks:List[Document],
    history:Optional[List[tuple[str,str]]] = None,
    provider:str = "stub",
    top_k:int =4,
    mmr_used:bool = False,
    config:Optional[PromptBuilderConfig] = None
) -> Dict[str,Any]:
    


    cfg = config or PromptBuilderConfig()

    context_budget = max(0,cfg.max_context_chars)
    history_budget = max(0,cfg.max_history_chars)

    trimmed_chunks = _reduce_context_to_budget(chunks,context_budget)

    history = history or []
    trimmed_history = _reduce_history_to_budget(history,history_budget)

    context_block = _format_context_inline(trimmed_chunks) if trimmed_chunks else "Context: (none)"
    history_block = _render_history(trimmed_history) if trimmed_history else "History: (none)"
    user_block = f"User question:\n{query or ''}".strip()

    system_text = f"{cfg.system_rules}\n{cfg.refusal_line}".strip()

    user_content = "\n\n".join([context_block,history_block,user_block])

    payload:Dict[str,Any] = {
        "system": system_text,
        "messages":[
            {
                "role":"user",
                "content":user_content,
            }
        ],
        "metadata":{
            "provider":provider,
            "top_k":int(top_k),
            "mmr_used":bool(mmr_used),
            "sources":_collect_sources(trimmed_chunks),
        }
    }
    return payload

# --- Compatibility layer expected by C1 / tests ---

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Keeps your original config name as the primary one
PromptConfig = PromptBuilderConfig  # alias for compatibility


@dataclass(frozen=True)
class PromptChunk:
    content: str
    source: str
    chunk_id: str


@dataclass(frozen=True)
class HistoryTurn:
    role: str  # "user" | "assistant"
    text: str


@dataclass(frozen=True)
class PromptInputs:
    query: str
    chunks: List[PromptChunk]
    history: List[HistoryTurn] = None
    top_k: int = 4
    provider: str = "stub"
    mmr_used: bool = False


class PromptBuilder:
    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptBuilderConfig()

    def build(self, inputs: PromptInputs) -> Dict[str, Any]:
        # Convert PromptChunk -> langchain Document for reuse of your build_prompt
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content=c.content,
                metadata={"source": c.source, "chunk_id": c.chunk_id},
            )
            for c in (inputs.chunks or [])
        ]

        # Convert HistoryTurn -> (user, assistant) pairs that your build_prompt accepts
        # We’ll compress consecutive roles into pairs: [(user_text, assistant_text), ...]
        pairs: List[tuple[str, str]] = []
        if inputs.history:
            buf_user = ""
            buf_assistant = ""
            for t in inputs.history:
                if t.role == "user":
                    # flush previous assistant if any
                    if buf_user or buf_assistant:
                        pairs.append((buf_user, buf_assistant))
                        buf_user, buf_assistant = "", ""
                    buf_user = t.text or ""
                elif t.role == "assistant":
                    buf_assistant = t.text or ""
                    # close the pair
                    pairs.append((buf_user, buf_assistant))
                    buf_user, buf_assistant = "", ""
            # tail if ended on a user turn
            if buf_user or buf_assistant:
                pairs.append((buf_user, buf_assistant))

        return build_prompt(
            query=inputs.query,
            chunks=docs,
            history=pairs,
            provider=inputs.provider,
            top_k=inputs.top_k,
            mmr_used=inputs.mmr_used,
            config=self.config,
        )