from typing import Optional,List,Dict,Any
import time
import json

class MemoryStore:
    def __init__(self, redis_client, max_turns:int = 6, ttl_seconds:Optional[int] = None):
        self.redis = redis_client
        self.max_turns = int(max_turns)
        self.ttl_seconds = int(ttl_seconds)
    
    @staticmethod
    def _key(tenant_id:str,session_id:str):
        return f"mem:{tenant_id}:{session_id}"
    
    def append_turn(
            self,
            tenant_id:str,
            session_id:str,
            role:str,
            text:str,
            ts:Optional[float] = None
    ) -> None:
        if role not in ("user","assistant"):
            raise ValueError("role must be either 'user' or 'assistant")
        ts = float(ts if ts is not None else time.time())

        key = self._key(tenant_id,session_id)
        payload = json.dumps({"role":role,"text":text or "","ts":ts})

        self.redis.rpush(key,payload)
        self.redis.ltrim(key,-self.max_turns,-1)

        if self.ttl_seconds:
            try:
                self.redis.expire(key,self.ttl_seconds)
            except Exception:
                pass
    
    def get_recent(self, tenant_id:str, session_id:str, limit:Optional[int]=None) ->List[Dict[str,Any]]:
        key = self._key(tenant_id,session_id)

        try:
            raw = self.redis.lrange(key,0,-1)
        except Exception:
            return []
        if not raw:
            return []
        
        items =[]
        for s in raw:
            try:
                items.append(json.loads(s))
            except Exception:
                continue
        if limit is not None and limit>=0:
            items = items[-limit:]
        
        return items
    
    def clear(self,tenant_id:str,session_id:str) -> None:
        key = self._key(tenant_id,session_id)
        try:
            self.redis.delete(key)
        except Exception:
            pass