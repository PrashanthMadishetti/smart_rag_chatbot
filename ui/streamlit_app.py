# streamlit_app.py
import os
import json
import requests
from typing import Dict, List, Optional, Any
import streamlit as st

# ====== Config ======
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TIMEOUT = 30
LS_KEY = "smart_rag_auth_v1"  # localStorage key

# LocalStorage bridge (requires: pip install streamlit-js-eval)
try:
    from streamlit_js_eval import get_local_storage, set_local_storage, remove_local_storage
except Exception:
    # Fallback stubs if the lib isn't installed yet (app will still run, but won't persist across refresh)
    def get_local_storage(key: str) -> Optional[str]:
        return None
    def set_local_storage(key: str, value: str) -> None:
        pass
    def remove_local_storage(key: str) -> None:
        pass


# ====== Session helpers & LS sync ======
def _auth_snapshot_from_state() -> dict:
    return {
        # Tokens
        "jwt":          st.session_state.get("jwt"),          # current token (login or tenant)
        "login_jwt":    st.session_state.get("login_jwt"),    # login-scoped token (used for switch-tenant)
        # Identity / scope
        "tenant_id":    st.session_state.get("tenant_id"),
        "tenant_name":  st.session_state.get("tenant_name"),
        "role":         st.session_state.get("role"),
        "email":        st.session_state.get("email"),
        "memberships":  st.session_state.get("memberships", []),
        # UI prefs
        "provider":     st.session_state.get("provider", "stub"),
        # Optional: persist chat so it doesn’t vanish on refresh (cleared on logout/switch)
        "messages":     st.session_state.get("messages", []),
    }

def _hydrate_state_from_snapshot(snap: dict) -> None:
    if not isinstance(snap, dict):
        return
    for k, v in snap.items():
        st.session_state[k] = v

def _write_auth_to_localstorage():
    try:
        set_local_storage(LS_KEY, json.dumps(_auth_snapshot_from_state()))
    except Exception:
        pass

def _read_auth_from_localstorage() -> Optional[dict]:
    try:
        raw = get_local_storage(LS_KEY)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None

def _clear_auth_localstorage():
    try:
        remove_local_storage(LS_KEY)
    except Exception:
        pass

def _ss_get(key: str, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def _ss_set(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v
    _write_auth_to_localstorage()  # keep LS in sync on every mutation

def _ss_reset_chat():
    st.session_state["messages"] = []
    _write_auth_to_localstorage()

def clear_auth_state():
    for k in ("jwt", "login_jwt", "tenant_id", "tenant_name", "role", "email", "memberships", "provider", "messages"):
        st.session_state.pop(k, None)
    _clear_auth_localstorage()

def ensure_message_buffer():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


# ====== API helpers ======
def api_get(path: str, token: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
    url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.get(url, headers=headers, params=params or {}, timeout=TIMEOUT)

def api_post(path: str, token: Optional[str] = None, json_body: Optional[Dict[str, Any]] = None, files=None, data=None):
    url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if files is None and data is None:
        headers["Content-Type"] = "application/json"
        return requests.post(url, headers=headers, json=json_body or {}, timeout=TIMEOUT)
    else:
        return requests.post(url, headers=headers, files=files, data=data, timeout=TIMEOUT)

def api_delete(path: str, token: Optional[str] = None):
    url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.delete(url, headers=headers, timeout=TIMEOUT)


# ====== Auth UI ======
def auth_panel():
    st.header("🔐 Sign in / Sign up")
    tabs = st.tabs(["Login", "Sign up"])

    # ---- Login tab ----
    with tabs[0]:
        with st.form("login_form", clear_on_submit=True):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")
        if submitted:
            if not email or not password:
                st.error("Email and password are required.")
            else:
                try:
                    r = api_post("/auth/login", json_body={"email": email, "password": password})
                    if r.status_code == 200:
                        data = r.json()
                        if "token" in data:
                            # Direct tenant token (rare path)
                            _ss_set(
                                jwt=data["token"],
                                login_jwt=None,           # we don't have a login-scoped token here
                                tenant_id=data.get("tenant_id"),
                                tenant_name=data.get("tenant_name"),
                                role=data.get("role"),
                                email=email,
                                memberships=[],           # no list in this path
                            )
                            ensure_message_buffer()
                            st.success("Logged in.")
                            st.rerun()
                        elif "login_token" in data:
                            # Normal path: keep login_jwt for future tenant switches
                            _ss_set(
                                jwt=data["login_token"],      # current token is the login token (until scoping)
                                login_jwt=data["login_token"],
                                tenant_id=None,
                                tenant_name=None,
                                role=None,
                                email=email,
                                memberships=data.get("memberships", []),
                            )
                            ensure_message_buffer()
                            st.info("Logged in. Pick a tenant from the sidebar to continue.")
                            st.rerun()
                        else:
                            st.warning("Unexpected response shape.")
                    else:
                        st.error(f"Login failed: {r.status_code} {r.text}")
                except Exception as e:
                    st.exception(e)

    # ---- Sign up tab ----
    with tabs[1]:
        with st.form("signup_form"):
            s_email = st.text_input("Email", key="signup_email")
            s_password = st.text_input("Password", type="password", key="signup_password")
            mode = st.radio("Choose onboarding mode", ["Create a new tenant", "Join an existing tenant"], key="signup_mode")

            tenant_name = ""
            tenant_join = ""
            joined_tenant_name = None

            if mode == "Create a new tenant":
                tenant_name = st.text_input("New tenant name", key="signup_tenant_name")
            else:
                st.write("Search or paste a tenant slug/ID.")
                q = st.text_input("Search tenants", key="tenant_search_q")
                tenant_list = []
                if q.strip():
                    resp = api_get("/auth/tenants/search", params={"q": q.strip()})
                    if resp.status_code == 200:
                        tenant_list = resp.json().get("tenants", [])
                    else:
                        st.warning(f"Search failed: {resp.status_code} {resp.text}")
                label_to_item = {f'{t["name"]} [{t["slug"]}]': t for t in tenant_list}
                options = list(label_to_item.keys()) or ["(no results)"]
                choice = st.selectbox("Choose a tenant", options=options, key="tenant_join_pick")
                if label_to_item and choice in label_to_item:
                    tenant_join = label_to_item[choice]["id"]
                    joined_tenant_name = label_to_item[choice]["name"]

            submitted = st.form_submit_button("Create account")
        if submitted:
            if not s_email or not s_password:
                st.error("Email and password are required.")
            else:
                body = {"email": s_email, "password": s_password}
                if mode == "Create a new tenant":
                    if not (tenant_name or "").strip():
                        st.error("Please provide a tenant name.")
                        return
                    body.update({"mode": "create_tenant", "tenant_name": tenant_name.strip()})
                else:
                    if not tenant_join:
                        st.error("Select a tenant to join.")
                        return
                    body.update({"mode": "join_tenant", "tenant_id": tenant_join})

                r = api_post("/auth/signup", json_body=body)
                if r.status_code == 200:
                    data = r.json()
                    effective_name = tenant_name.strip() if mode == "Create a new tenant" else (joined_tenant_name or None)
                    _ss_set(
                        jwt=data["token"],            # tenant-scoped token
                        login_jwt=None,               # signup flow doesn’t issue login token
                        tenant_id=data.get("tenant_id"),
                        tenant_name=effective_name,
                        role=data.get("role"),
                        email=s_email,
                        memberships=[],               # scoped already
                    )
                    _ss_reset_chat()
                    st.success("Account created and tenant selected.")
                    st.rerun()
                else:
                    st.error(f"Signup failed: {r.status_code} {r.text}")


# ====== Sidebar: Tenant & Provider (post-login) ======
def tenant_sidebar():
    # Rehydrate provider default to keep selectbox index valid
    if _ss_get("provider") is None:
        _ss_set(provider="stub")

    jwt = _ss_get("jwt")
    if not jwt:
        st.sidebar.info("Log in to pick a tenant.")
        return

    st.sidebar.markdown("### 🏷️ Tenant")

    # Provider selector (only when authenticated)
    st.sidebar.markdown("#### LLM Provider")
    provider_options = ["stub", "gemini", "groq"]
    current_provider = _ss_get("provider") or "stub"
    try:
        idx = provider_options.index(current_provider)
    except ValueError:
        idx = 0
    provider = st.sidebar.selectbox("Choose provider", provider_options, index=idx, key="provider_select")
    _ss_set(provider=provider)

    # Current tenant badge
    current_tenant_id = _ss_get("tenant_id")
    current_tenant_name = _ss_get("tenant_name")
    if current_tenant_name:
        st.sidebar.success(f"Current: {current_tenant_name} ({_ss_get('role') or '-'})")
    elif current_tenant_id:
        st.sidebar.success(f"Current: {current_tenant_id} ({_ss_get('role') or '-'})")
    else:
        st.sidebar.warning("No tenant selected.")

    # Switch / Create
    memberships = _ss_get("memberships", []) or []
    options = []
    value_to_item = {}
    for m in memberships:
        label = f'{m["tenant_name"]} ({m.get("role","")})'
        options.append(label)
        value_to_item[label] = m
    options.append("➕ Create new tenant…")

    choice = st.sidebar.selectbox("Switch or create", options, key="tenant_pick_sidebar")

    if choice == "➕ Create new tenant…":
        new_name = st.sidebar.text_input("New tenant name", key="new_tenant_name_sidebar")
        if st.sidebar.button("Create"):
            nm = (new_name or "").strip()
            if not nm:
                st.sidebar.error("Tenant name required")
            else:
                # /auth/tenants/create accepts login or tenant token in your updated backend
                r = api_post("/auth/tenants/create", token=_ss_get("jwt"), json_body={"name": nm})
                if r.status_code == 200:
                    data = r.json()
                    # Switch to the new tenant immediately
                    _ss_set(
                        jwt=data["token"],                         # tenant token for new tenant
                        tenant_id=data["tenant_id"],
                        tenant_name=data.get("tenant_name", nm),
                        role=data.get("role", "owner"),
                    )
                    # Keep the login_jwt if we have it, so user can still switch later
                    ms = _ss_get("memberships", []) or []
                    ms.append({
                        "tenant_id": data["tenant_id"],
                        "tenant_name": data.get("tenant_name", nm),
                        "role": "owner",
                    })
                    _ss_set(memberships=ms)
                    _ss_reset_chat()
                    st.sidebar.success(f"Created & switched to {data.get('tenant_name', nm)}")
                    st.rerun()
                else:
                    st.sidebar.error(f"Create failed: {r.status_code} {r.text}")
    else:
        chosen = value_to_item.get(choice)
        if chosen:
            # We need the LOGIN token to switch tenants
            login_jwt = _ss_get("login_jwt")
            disabled = login_jwt is None
            help_txt = None
            if disabled:
                help_txt = "To switch tenants, please log out and log back in (we need a login token)."

            if st.sidebar.button("Switch Tenant", disabled=disabled, help=help_txt):
                r = api_post("/auth/switch-tenant", token=login_jwt, json_body={"tenant_id": chosen["tenant_id"]})
                if r.status_code == 200:
                    data = r.json()
                    _ss_set(
                        jwt=data["token"],                 # new tenant-scoped token
                        tenant_id=data["tenant_id"],
                        tenant_name=chosen["tenant_name"],
                        role=data.get("role"),
                    )
                    _ss_reset_chat()
                    st.sidebar.success(f"Switched to {chosen['tenant_name']}")
                    st.rerun()
                else:
                    st.sidebar.error(f"Switch tenant failed: {r.status_code} {r.text}")


# ====== Header ======
def header_bar():
    jwt = _ss_get("jwt")
    tenant_name = _ss_get("tenant_name")
    role = _ss_get("role")
    email = _ss_get("email")
    left, right = st.columns([0.7, 0.3])
    with left:
        st.caption("Smart RAG Chatbot — Demo UI")
    with right:
        if jwt:
            if tenant_name:
                st.success(f"Tenant: {tenant_name}  |  Role: {role or '-'}")
            else:
                st.warning("Tenant: (not selected)")
            st.caption(f"Signed in as {email or '-'}")
            if st.button("Log out"):
                clear_auth_state()
                st.rerun()
        else:
            st.warning("Not authenticated")


# ====== Ingest UI ======
def ingest_panel():
    st.header("📥 Ingest")
    jwt = _ss_get("jwt")
    tenant_id = _ss_get("tenant_id")

    if not jwt:
        st.info("Please log in first.")
        return
    if not tenant_id:
        st.info("Pick a tenant from the sidebar before ingesting.")
        return

    st.subheader("Upload a file")
    f = st.file_uploader("PDF or TXT", type=["pdf", "txt"], key="upload_file")
    if st.button("Ingest File"):
        if f is None:
            st.error("Choose a file first.")
        else:
            files = {"file": (f.name, f.getvalue(), getattr(f, "type", "application/octet-stream"))}
            r = api_post("/ingest", token=jwt, files=files)
            if r.status_code == 200:
                st.success(f"Ingested: {r.json()}")
            else:
                st.error(f"Upload failed: {r.status_code} {r.text}")

    st.subheader("Ingest URLs")
    urls_csv = st.text_area("Enter one or more URLs (comma or newline separated)")
    if st.button("Ingest URLs"):
        urls = [u.strip() for u in urls_csv.replace("\n", ",").split(",") if u.strip()]
        if not urls:
            st.error("Provide at least one URL.")
        else:
            r = api_post("/ingest", token=jwt, json_body={"urls": urls})
            if r.status_code == 200:
                st.success(f"Ingested: {r.json()}")
            else:
                st.error(f"URL ingest failed: {r.status_code} {r.text}")


# ====== Documents UI ======
def documents_panel():
    st.header("📄 Documents")
    jwt = _ss_get("jwt")
    tenant_id = _ss_get("tenant_id")

    if not jwt:
        st.info("Please log in first.")
        return
    if not tenant_id:
        st.info("Pick a tenant from the sidebar first.")
        return

    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        if st.button("🔁 Refresh"):
            st.rerun()

    # Fetch list
    r = api_get("/documents", token=jwt)
    # print("DOCS STATUS:", r.status_code)
    # print("DOCS BODY:", r.text[:2000])
    if r.status_code != 200:
        st.error(f"Failed to load: {r.status_code} {r.text}")
        return

    items = r.json().get("items", [])
    if not items:
        st.info("No documents found for this tenant yet.")
        return

    for doc in items:
        # with st.expander(f'{doc["title"]}  •  {doc["source"]}'):
        with st.expander(f' {doc["source"]}'):
            st.write(f'**Chunks:** {doc["chunk_count"]}')
            st.caption(f'Created: {doc["created_at"]}')
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ Delete", key=f"del_{doc['id']}"):
                    rr = api_delete(f"/documents/{doc['id']}", token=jwt)
                    if rr.status_code == 200:
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {rr.status_code} {rr.text}")
            with c2:
                if st.button("♻️ Reindex (stub)", key=f"re_{doc['id']}"):
                    rr = api_post(f"/documents/reindex/{doc['id']}", token=jwt, json_body={})
                    if rr.status_code == 200:
                        st.success("Reindex request accepted.")
                    else:
                        st.error(f"Reindex failed: {rr.status_code} {rr.text}")


# ====== Chat UI ======
def chat_panel():
    st.header("💬 Chat")
    jwt = _ss_get("jwt")
    tenant_id = _ss_get("tenant_id")
    if not jwt:
        st.info("Please log in first.")
        return
    if not tenant_id:
        st.info("Pick a tenant from the sidebar before chatting.")
        return

    ensure_message_buffer()

    # Render history
    for m in st.session_state["messages"]:
        role = m.get("role", "user")
        with st.chat_message(role):
            st.write(m.get("text", ""))

    provider = _ss_get("provider", "stub")

    prompt = st.chat_input("Ask a question")
    if prompt:
        st.session_state["messages"].append({"role": "user", "text": prompt})
        _write_auth_to_localstorage()  # persist chat after user message
        with st.chat_message("user"):
            st.write(prompt)

        body = {
            "session_id": "demo-session",
            "question": prompt,
            "k": 4,
            "provider": provider,
            "use_mmr": True,
            "mmr_lambda": 0.7,
        }
        with st.chat_message("assistant"):
            try:
                r = api_post("/query", token=jwt, json_body=body)
                if r.status_code == 200:
                    data = r.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    st.write(answer)
                    if sources:
                        st.caption("Sources: " + ", ".join(sources))
                    st.session_state["messages"].append({"role": "assistant", "text": answer})
                    _write_auth_to_localstorage()  # persist chat after assistant reply
                elif r.status_code == 401:
                    st.error("Unauthorized — your token might be missing or expired.")
                else:
                    st.error(f"Query failed: {r.status_code} {r.text}")
            except Exception as e:
                st.exception(e)


# ====== Main ======
def main():
    st.set_page_config(page_title="Smart RAG Chatbot", page_icon="🤖", layout="wide")

    # Rehydrate once per new Streamlit session
    if not st.session_state.get("_rehydrated_from_ls"):
        snap = _read_auth_from_localstorage()
        if isinstance(snap, dict) and (snap.get("jwt") or snap.get("login_jwt")):
            _hydrate_state_from_snapshot(snap)
        # Ensure defaults exist
        if _ss_get("provider") is None:
            _ss_set(provider="stub")
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        st.session_state["_rehydrated_from_ls"] = True

    # Sidebar tenant controls (post-login)
    tenant_sidebar()

    # Top header
    header_bar()

    # If not authenticated, show auth panel only
    if not _ss_get("jwt"):
        auth_panel()
        return

    # Authenticated: show sections
    tab = st.sidebar.radio("Navigate", ["Chat", "Ingest", "Documents", "Account"])
    if tab == "Chat":
        chat_panel()
    elif tab == "Ingest":
        ingest_panel()
    elif tab == "Documents":
        documents_panel()
    else:
        st.header("👤 Account")
        st.write(f"Email: {_ss_get('email') or '-'}")
        st.write(f"Tenant: {_ss_get('tenant_name') or '(not selected)'}")
        st.write(f"Role: {_ss_get('role') or '-'}")
        if st.button("Log out (clear token)"):
            clear_auth_state()
            st.rerun()


if __name__ == "__main__":
    main()

# import os
# import json
# import requests
# from typing import Dict, List, Optional, Any
# import streamlit as st

# # ====== Config ======
# API_BASE = os.getenv("API_BASE", "http://localhost:8000")
# TIMEOUT = 30
# LS_KEY = "smart_rag_auth_v1"  # localStorage key

# # LocalStorage bridge
# try:
#     from streamlit_js_eval import get_local_storage, set_local_storage, remove_local_storage
# except Exception:
#     # Fallback stubs if the lib isn't installed yet (app will still run, but won't persist across refresh)
#     def get_local_storage(key: str) -> Optional[str]:
#         return None
#     def set_local_storage(key: str, value: str) -> None:
#         pass
#     def remove_local_storage(key: str) -> None:
#         pass


# # ====== Session helpers & LS sync ======
# def _auth_snapshot_from_state() -> dict:
#     return {
#         # Tokens
#         "jwt":          st.session_state.get("jwt"),          # current token (login or tenant)
#         "login_jwt":    st.session_state.get("login_jwt"),    # login-scoped token (used for switch-tenant)
#         # Identity / scope
#         "tenant_id":    st.session_state.get("tenant_id"),
#         "tenant_name":  st.session_state.get("tenant_name"),
#         "role":         st.session_state.get("role"),
#         "email":        st.session_state.get("email"),
#         "memberships":  st.session_state.get("memberships", []),
#         # UI prefs
#         "provider":     st.session_state.get("provider", "stub"),
#         # Optional: persist chat so it doesn’t vanish on refresh (cleared on logout/switch)
#         "messages":     st.session_state.get("messages", []),
#     }

# def _hydrate_state_from_snapshot(snap: dict) -> None:
#     if not isinstance(snap, dict):
#         return
#     for k, v in snap.items():
#         st.session_state[k] = v

# def _write_auth_to_localstorage():
#     try:
#         set_local_storage(LS_KEY, json.dumps(_auth_snapshot_from_state()))
#     except Exception:
#         pass

# def _read_auth_from_localstorage() -> Optional[dict]:
#     try:
#         raw = get_local_storage(LS_KEY)
#         if raw:
#             return json.loads(raw)
#     except Exception:
#         pass
#     return None

# def _clear_auth_localstorage():
#     try:
#         remove_local_storage(LS_KEY)
#     except Exception:
#         pass

# def _ss_get(key: str, default=None):
#     if key not in st.session_state:
#         st.session_state[key] = default
#     return st.session_state[key]

# def _ss_set(**kwargs):
#     for k, v in kwargs.items():
#         st.session_state[k] = v
#     _write_auth_to_localstorage()  # keep LS in sync on every mutation

# def _ss_reset_chat():
#     st.session_state["messages"] = []
#     _write_auth_to_localstorage()

# def clear_auth_state():
#     for k in ("jwt", "login_jwt", "tenant_id", "tenant_name", "role", "email", "memberships", "provider", "messages"):
#         st.session_state.pop(k, None)
#     _clear_auth_localstorage()

# def ensure_message_buffer():
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []


# # ====== API helpers ======
# def api_get(path: str, token: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
#     url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
#     headers = {}
#     if token:
#         headers["Authorization"] = f"Bearer {token}"
#     return requests.get(url, headers=headers, params=params or {}, timeout=TIMEOUT)

# def api_post(path: str, token: Optional[str] = None, json_body: Optional[Dict[str, Any]] = None, files=None, data=None):
#     url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
#     headers = {}
#     if token:
#         headers["Authorization"] = f"Bearer {token}"
#     if files is None and data is None:
#         headers["Content-Type"] = "application/json"
#         return requests.post(url, headers=headers, json=json_body or {}, timeout=TIMEOUT)
#     else:
#         return requests.post(url, headers=headers, files=files, data=data, timeout=TIMEOUT)

# def api_delete(path: str, token: Optional[str] = None):
#     url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
#     headers = {}
#     if token:
#         headers["Authorization"] = f"Bearer {token}"
#     return requests.delete(url, headers=headers, timeout=TIMEOUT)

# # ====== Auth UI ======
# def auth_panel():
#     st.header("🔐 Sign in / Sign up")
#     tabs = st.tabs(["Login", "Sign up"])

#     # ---- Login tab ----
#     with tabs[0]:
#         with st.form("login_form", clear_on_submit=True):
#             email = st.text_input("Email", key="login_email")
#             password = st.text_input("Password", type="password", key="login_password")
#             submitted = st.form_submit_button("Login")
#         if submitted:
#             if not email or not password:
#                 st.error("Email and password are required.")
#             else:
#                 try:
#                     r = api_post("/auth/login", json_body={"email": email, "password": password})
#                     if r.status_code == 200:
#                         data = r.json()
#                         if "token" in data:
#                             # Direct tenant token (rare path)
#                             _ss_set(
#                                 jwt=data["token"],
#                                 login_jwt=None,           # we don't have a login-scoped token here
#                                 tenant_id=data.get("tenant_id"),
#                                 tenant_name=data.get("tenant_name"),
#                                 role=data.get("role"),
#                                 email=email,
#                                 memberships=[],           # no list in this path
#                             )
#                             ensure_message_buffer()
#                             st.success("Logged in.")
#                             st.rerun()
#                         elif "login_token" in data:
#                             # Normal path: keep login_jwt for future tenant switches
#                             _ss_set(
#                                 jwt=data["login_token"],      # current token is the login token (until scoping)
#                                 login_jwt=data["login_token"],
#                                 tenant_id=None,
#                                 tenant_name=None,
#                                 role=None,
#                                 email=email,
#                                 memberships=data.get("memberships", []),
#                             )
#                             ensure_message_buffer()
#                             st.info("Logged in. Pick a tenant from the sidebar to continue.")
#                             st.rerun()
#                         else:
#                             st.warning("Unexpected response shape.")
#                     else:
#                         st.error(f"Login failed: {r.status_code} {r.text}")
#                 except Exception as e:
#                     st.exception(e)

#     # ---- Sign up tab ----
#     with tabs[1]:
#         with st.form("signup_form"):
#             s_email = st.text_input("Email", key="signup_email")
#             s_password = st.text_input("Password", type="password", key="signup_password")
#             mode = st.radio("Choose onboarding mode", ["Create a new tenant", "Join an existing tenant"], key="signup_mode")

#             tenant_name = ""
#             tenant_join = ""
#             joined_tenant_name = None

#             if mode == "Create a new tenant":
#                 tenant_name = st.text_input("New tenant name", key="signup_tenant_name")
#             else:
#                 st.write("Search or paste a tenant slug/ID.")
#                 q = st.text_input("Search tenants", key="tenant_search_q")
#                 tenant_list = []
#                 if q.strip():
#                     resp = api_get("/auth/tenants/search", params={"q": q.strip()})
#                     if resp.status_code == 200:
#                         tenant_list = resp.json().get("tenants", [])
#                     else:
#                         st.warning(f"Search failed: {resp.status_code} {resp.text}")
#                 label_to_item = {f'{t["name"]} [{t["slug"]}]': t for t in tenant_list}
#                 options = list(label_to_item.keys()) or ["(no results)"]
#                 choice = st.selectbox("Choose a tenant", options=options, key="tenant_join_pick")
#                 if label_to_item and choice in label_to_item:
#                     tenant_join = label_to_item[choice]["id"]
#                     joined_tenant_name = label_to_item[choice]["name"]

#             submitted = st.form_submit_button("Create account")
#         if submitted:
#             if not s_email or not s_password:
#                 st.error("Email and password are required.")
#             else:
#                 body = {"email": s_email, "password": s_password}
#                 if mode == "Create a new tenant":
#                     if not (tenant_name or "").strip():
#                         st.error("Please provide a tenant name.")
#                         return
#                     body.update({"mode": "create_tenant", "tenant_name": tenant_name.strip()})
#                 else:
#                     if not tenant_join:
#                         st.error("Select a tenant to join.")
#                         return
#                     body.update({"mode": "join_tenant", "tenant_id": tenant_join})

#                 r = api_post("/auth/signup", json_body=body)
#                 if r.status_code == 200:
#                     data = r.json()
#                     effective_name = tenant_name.strip() if mode == "Create a new tenant" else (joined_tenant_name or None)
#                     _ss_set(
#                         jwt=data["token"],            # tenant-scoped token
#                         login_jwt=None,               # signup flow doesn’t issue login token
#                         tenant_id=data.get("tenant_id"),
#                         tenant_name=effective_name,
#                         role=data.get("role"),
#                         email=s_email,
#                         memberships=[],               # scoped already
#                     )
#                     _ss_reset_chat()
#                     st.success("Account created and tenant selected.")
#                     st.rerun()
#                 else:
#                     st.error(f"Signup failed: {r.status_code} {r.text}")


# # ====== Sidebar: Tenant & Provider (post-login) ======
# def tenant_sidebar():
#     # Rehydrate provider default to keep selectbox index valid
#     if _ss_get("provider") is None:
#         _ss_set(provider="stub")

#     jwt = _ss_get("jwt")
#     if not jwt:
#         st.sidebar.info("Log in to pick a tenant.")
#         return

#     st.sidebar.markdown("### 🏷️ Tenant")

#     # Provider selector (only when authenticated)
#     st.sidebar.markdown("#### LLM Provider")
#     provider_options = ["stub", "gemini", "groq"]
#     current_provider = _ss_get("provider") or "stub"
#     try:
#         idx = provider_options.index(current_provider)
#     except ValueError:
#         idx = 0
#     provider = st.sidebar.selectbox("Choose provider", provider_options, index=idx, key="provider_select")
#     _ss_set(provider=provider)

#     # Current tenant badge
#     current_tenant_id = _ss_get("tenant_id")
#     current_tenant_name = _ss_get("tenant_name")
#     if current_tenant_name:
#         st.sidebar.success(f"Current: {current_tenant_name} ({_ss_get('role') or '-'})")
#     elif current_tenant_id:
#         st.sidebar.success(f"Current: {current_tenant_id} ({_ss_get('role') or '-'})")
#     else:
#         st.sidebar.warning("No tenant selected.")

#     # Switch / Create
#     memberships = _ss_get("memberships", []) or []
#     options = []
#     value_to_item = {}
#     for m in memberships:
#         label = f'{m["tenant_name"]} ({m.get("role","")})'
#         options.append(label)
#         value_to_item[label] = m
#     options.append("➕ Create new tenant…")

#     choice = st.sidebar.selectbox("Switch or create", options, key="tenant_pick_sidebar")

#     if choice == "➕ Create new tenant…":
#         new_name = st.sidebar.text_input("New tenant name", key="new_tenant_name_sidebar")
#         if st.sidebar.button("Create"):
#             nm = (new_name or "").strip()
#             if not nm:
#                 st.sidebar.error("Tenant name required")
#             else:
#                 # /auth/tenants/create accepts login or tenant token in your updated backend
#                 r = api_post("/auth/tenants/create", token=_ss_get("jwt"), json_body={"name": nm})
#                 if r.status_code == 200:
#                     data = r.json()
#                     # Switch to the new tenant immediately
#                     _ss_set(
#                         jwt=data["token"],                         # tenant token for new tenant
#                         tenant_id=data["tenant_id"],
#                         tenant_name=data.get("tenant_name", nm),
#                         role=data.get("role", "owner"),
#                     )
#                     # Keep the login_jwt if we have it, so user can still switch later
#                     ms = _ss_get("memberships", []) or []
#                     ms.append({
#                         "tenant_id": data["tenant_id"],
#                         "tenant_name": data.get("tenant_name", nm),
#                         "role": "owner",
#                     })
#                     _ss_set(memberships=ms)
#                     _ss_reset_chat()
#                     st.sidebar.success(f"Created & switched to {data.get('tenant_name', nm)}")
#                     st.rerun()
#                 else:
#                     st.sidebar.error(f"Create failed: {r.status_code} {r.text}")
#     else:
#         chosen = value_to_item.get(choice)
#         if chosen:
#             # We need the LOGIN token to switch tenants
#             login_jwt = _ss_get("login_jwt")
#             disabled = login_jwt is None
#             help_txt = None
#             if disabled:
#                 help_txt = "To switch tenants, please log out and log back in (we need a login token)."

#             if st.sidebar.button("Switch Tenant", disabled=disabled, help=help_txt):
#                 r = api_post("/auth/switch-tenant", token=login_jwt, json_body={"tenant_id": chosen["tenant_id"]})
#                 if r.status_code == 200:
#                     data = r.json()
#                     _ss_set(
#                         jwt=data["token"],                 # new tenant-scoped token
#                         tenant_id=data["tenant_id"],
#                         tenant_name=chosen["tenant_name"],
#                         role=data.get("role"),
#                     )
#                     _ss_reset_chat()
#                     st.sidebar.success(f"Switched to {chosen['tenant_name']}")
#                     st.rerun()
#                 else:
#                     st.sidebar.error(f"Switch tenant failed: {r.status_code} {r.text}")


# # ====== Header ======
# def header_bar():
#     jwt = _ss_get("jwt")
#     tenant_name = _ss_get("tenant_name")
#     role = _ss_get("role")
#     email = _ss_get("email")
#     left, right = st.columns([0.7, 0.3])
#     with left:
#         st.caption("Smart RAG Chatbot — Demo UI")
#     with right:
#         if jwt:
#             if tenant_name:
#                 st.success(f"Tenant: {tenant_name}  |  Role: {role or '-'}")
#             else:
#                 st.warning("Tenant: (not selected)")
#             st.caption(f"Signed in as {email or '-'}")
#             if st.button("Log out"):
#                 clear_auth_state()
#                 st.rerun()
#         else:
#             st.warning("Not authenticated")


# # ====== Ingest UI ======
# def ingest_panel():
#     st.header("📥 Ingest")
#     jwt = _ss_get("jwt")
#     tenant_id = _ss_get("tenant_id")

#     if not jwt:
#         st.info("Please log in first.")
#         return
#     if not tenant_id:
#         st.info("Pick a tenant from the sidebar before ingesting.")
#         return

#     st.subheader("Upload a file")
#     f = st.file_uploader("PDF or TXT", type=["pdf", "txt"], key="upload_file")
#     if st.button("Ingest File"):
#         if f is None:
#             st.error("Choose a file first.")
#         else:
#             files = {"file": (f.name, f.getvalue(), getattr(f, "type", "application/octet-stream"))}
#             r = api_post("/ingest", token=jwt, files=files)
#             if r.status_code == 200:
#                 st.success(f"Ingested: {r.json()}")
#             else:
#                 st.error(f"Upload failed: {r.status_code} {r.text}")

#     st.subheader("Ingest URLs")
#     urls_csv = st.text_area("Enter one or more URLs (comma or newline separated)")
#     if st.button("Ingest URLs"):
#         urls = [u.strip() for u in urls_csv.replace("\n", ",").split(",") if u.strip()]
#         if not urls:
#             st.error("Provide at least one URL.")
#         else:
#             r = api_post("/ingest", token=jwt, json_body={"urls": urls})
#             if r.status_code == 200:
#                 st.success(f"Ingested: {r.json()}")
#             else:
#                 st.error(f"URL ingest failed: {r.status_code} {r.text}")

# def documents_panel():
#     st.header("📄 Documents")
#     jwt = _ss_get("jwt")
#     tenant_id = _ss_get("tenant_id")

#     if not jwt:
#         st.info("Please log in first.")
#         return
#     if not tenant_id:
#         st.info("Pick a tenant from the sidebar first.")
#         return

#     col1, col2 = st.columns([0.2, 0.8])
#     with col1:
#         if st.button("🔁 Refresh"):
#             st.experimental_rerun()

#     # Fetch list
#     r = api_get("/docs", token=jwt)
#     if r.status_code != 200:
#         st.error(f"Failed to load: {r.status_code} {r.text}")
#         return
#     print(r)
#     items = r.json().get("items", [])
#     if not items:
#         st.info("No documents found for this tenant yet.")
#         return

#     for doc in items:
#         with st.expander(f'{doc["title"]}  •  {doc["source"]}'):
#             st.write(f'**Chunks:** {doc["chunk_count"]}')
#             st.caption(f'Created: {doc["created_at"]}')
#             c1, c2 = st.columns(2)
#             with c1:
#                 if st.button("🗑️ Delete", key=f"del_{doc['id']}"):
#                     rr = api_delete(f"/docs/{doc['id']}", token=jwt)
#                     if rr.status_code == 200:
#                         st.success("Deleted.")
#                         st.experimental_rerun()
#                     else:
#                         st.error(f"Delete failed: {rr.status_code} {rr.text}")
#             with c2:
#                 if st.button("♻️ Reindex (stub)", key=f"re_{doc['id']}"):
#                     rr = api_post(f"/docs/reindex/{doc['id']}", token=jwt, json_body={})
#                     if rr.status_code == 200:
#                         st.success("Reindex request accepted.")
#                     else:
#                         st.error(f"Reindex failed: {rr.status_code} {rr.text}")

# # ====== Chat UI ======
# def chat_panel():
#     st.header("💬 Chat")
#     jwt = _ss_get("jwt")
#     tenant_id = _ss_get("tenant_id")
#     if not jwt:
#         st.info("Please log in first.")
#         return
#     if not tenant_id:
#         st.info("Pick a tenant from the sidebar before chatting.")
#         return

#     ensure_message_buffer()

#     # Render history
#     for m in st.session_state["messages"]:
#         role = m.get("role", "user")
#         with st.chat_message(role):
#             st.write(m.get("text", ""))

#     provider = _ss_get("provider", "stub")

#     prompt = st.chat_input("Ask a question")
#     if prompt:
#         st.session_state["messages"].append({"role": "user", "text": prompt})
#         _write_auth_to_localstorage()  # persist chat after user message
#         with st.chat_message("user"):
#             st.write(prompt)

#         body = {
#             "session_id": "demo-session",
#             "question": prompt,
#             "k": 4,
#             "provider": provider,
#             "use_mmr": True,
#             "mmr_lambda": 0.7,
#         }
#         with st.chat_message("assistant"):
#             try:
#                 r = api_post("/query", token=jwt, json_body=body)
#                 if r.status_code == 200:
#                     data = r.json()
#                     answer = data.get("answer", "")
#                     sources = data.get("sources", [])
#                     st.write(answer)
#                     if sources:
#                         st.caption("Sources: " + ", ".join(sources))
#                     st.session_state["messages"].append({"role": "assistant", "text": answer})
#                     _write_auth_to_localstorage()  # persist chat after assistant reply
#                 elif r.status_code == 401:
#                     st.error("Unauthorized — your token might be missing or expired.")
#                 else:
#                     st.error(f"Query failed: {r.status_code} {r.text}")
#             except Exception as e:
#                 st.exception(e)


# # ====== Main ======
# def main():
#     st.set_page_config(page_title="Smart RAG Chatbot", page_icon="🤖", layout="wide")

#     # Rehydrate once per new Streamlit session
#     if not st.session_state.get("_rehydrated_from_ls"):
#         snap = _read_auth_from_localstorage()
#         if isinstance(snap, dict) and (snap.get("jwt") or snap.get("login_jwt")):
#             _hydrate_state_from_snapshot(snap)
#         # Ensure defaults exist
#         if _ss_get("provider") is None:
#             _ss_set(provider="stub")
#         if "messages" not in st.session_state:
#             st.session_state["messages"] = []
#         st.session_state["_rehydrated_from_ls"] = True

#     # Sidebar tenant controls (post-login)
#     tenant_sidebar()

#     # Top header
#     header_bar()

#     # If not authenticated, show auth panel only
#     if not _ss_get("jwt"):
#         auth_panel()
#         return

#     # Authenticated: show sections
#     tab = st.sidebar.radio("Navigate", ["Chat", "Ingest", "Account","Documents"])
#     if tab == "Chat":
#         chat_panel()
#     elif tab == "Ingest":
#         ingest_panel()
#     elif tab == "Documents":
#         documents_panel()
#     else:
#         st.header("👤 Account")
#         st.write(f"Email: {_ss_get('email') or '-'}")
#         st.write(f"Tenant: {_ss_get('tenant_name') or '(not selected)'}")
#         st.write(f"Role: {_ss_get('role') or '-'}")
#         if st.button("Log out (clear token)"):
#             clear_auth_state()
#             st.rerun()


# if __name__ == "__main__":
#     main()