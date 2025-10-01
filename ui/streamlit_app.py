# streamlit_app.py
import os
import json
import requests
from typing import Dict, List, Optional, Any
import streamlit as st

# ---------- Config ----------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TIMEOUT = 30

# ---------- Session Helpers ----------
def _ss_get(key: str, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def _ss_set(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v

def clear_auth_state():
    for k in ("jwt", "tenant_id", "tenant_name", "role", "email", "memberships"):
        st.session_state.pop(k, None)

def ensure_message_buffer():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list of {"role":"user|assistant", "text":"..."}

# ---------- API Helpers ----------
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

# ---------- UI: Auth ----------
def auth_panel():
    st.header("🔐 Sign in / Sign up")
    tabs = st.tabs(["Login", "Sign up"])

    # ---- Login tab (no tenant selection here) ----
    with tabs[0]:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")
        if submitted:
            if not email or not password:
                st.error("Email and password are required.")
            else:
                try:
                    payload = {"email": email, "password": password}
                    r = api_post("/auth/login", json_body=payload)
                    if r.status_code == 200:
                        data = r.json()
                        # A) tenant token directly (rare here)
                        if "token" in data:
                            _ss_set(
                                jwt=data["token"],
                                tenant_id=data.get("tenant_id"),
                                tenant_name=data.get("tenant_name"),  # may or may not be present
                                role=data.get("role"),
                                email=email,
                                memberships=[],
                            )
                            st.success("Logged in.")
                        # B) login token + memberships (choose from sidebar)
                        elif "login_token" in data:
                            _ss_set(
                                jwt=data["login_token"],
                                tenant_id=None,
                                tenant_name=None,
                                role=None,
                                email=email,
                                memberships=data.get("memberships", []),  # [{tenant_id, tenant_name, role}]
                            )
                            st.info("Logged in. Pick a tenant from the sidebar to continue.")
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

            mode = st.radio(
                "Choose onboarding mode",
                ["Create a new tenant", "Join an existing tenant"],
                key="signup_mode"
            )
            tenant_name = ""
            tenant_join = ""
            joined_tenant_name = None

            if mode == "Create a new tenant":
                tenant_name = st.text_input("New tenant name", key="signup_tenant_name")
            else:
                st.write("Search or paste a tenant slug/ID. (Enumeration allowed)")
                q = st.text_input("Search tenants", key="tenant_search_q")
                tenant_list = []
                if q.strip():
                    resp = api_get("/auth/tenants/search", params={"q": q.strip()})
                    if resp.status_code == 200:
                        tenant_list = resp.json().get("tenants", [])
                    else:
                        st.warning(f"Search failed: {resp.status_code} {resp.text}")
                # keep a mapping to resolve name for display state later
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
                    if not tenant_name.strip():
                        st.error("Please provide a tenant name.")
                    else:
                        body.update({"mode": "create_tenant", "tenant_name": tenant_name.strip()})
                else:
                    if not tenant_join:
                        st.error("Select a tenant to join.")
                    else:
                        body.update({"mode": "join_tenant", "tenant_id": tenant_join})

                r = api_post("/auth/signup", json_body=body)
                if r.status_code == 200:
                    data = r.json()
                    # Store name as well for UI (created or joined)
                    effective_name = tenant_name.strip() if mode == "Create a new tenant" else (joined_tenant_name or None)
                    _ss_set(
                        jwt=data["token"],
                        tenant_id=data.get("tenant_id"),
                        tenant_name=effective_name,
                        role=data.get("role"),
                        email=s_email,
                        memberships=[],
                    )
                    st.success("Account created and tenant selected.")
                else:
                    st.error(f"Signup failed: {r.status_code} {r.text}")

# ---------- Sidebar: Tenant selection AFTER login ----------
def tenant_sidebar():
    jwt = _ss_get("jwt")
    if not jwt:
        st.sidebar.info("Log in to pick a tenant.")
        return

    st.sidebar.markdown("### 🏷️ Tenant")
    current_tenant_id = _ss_get("tenant_id")
    current_tenant_name = _ss_get("tenant_name")
    memberships = _ss_get("memberships", [])

    if current_tenant_name:
        st.sidebar.success(f"Current: {current_tenant_name} ({_ss_get('role') or '-'})")
    elif current_tenant_id:
        # Fallback (shouldn’t happen often)
        st.sidebar.success(f"Current: {current_tenant_id} ({_ss_get('role') or '-'})")
    else:
        st.sidebar.warning("No tenant selected.")

    # If we have memberships from login, offer a picker to switch
    if memberships:
        # Build label -> item map
        # memberships items: {"tenant_id", "tenant_name", "role"}
        label_to_item = {
            f'{m["tenant_name"]} ({m.get("role","")}) [{m["tenant_id"]}]': m
            for m in memberships
        }
        if label_to_item:
            choice = st.sidebar.selectbox(
                "Switch tenant",
                list(label_to_item.keys()),
                key="tenant_pick_sidebar",
            )
            if st.sidebar.button("Switch Tenant"):
                chosen = label_to_item[choice]
                chosen_id = chosen["tenant_id"]
                chosen_name = chosen["tenant_name"]
                r = api_post("/auth/switch-tenant", token=jwt, json_body={"tenant_id": chosen_id})
                if r.status_code == 200:
                    data = r.json()
                    # Now we have a tenant-scoped token; persist both id and name
                    _ss_set(
                        jwt=data["token"],
                        tenant_id=data["tenant_id"],
                        tenant_name=chosen_name,   # <-- keep friendly name locally
                        role=data.get("role"),
                    )
                    # Once scoped, memberships list can be cleared (optional)
                    st.session_state.pop("memberships", None)
                    st.sidebar.success(f"Switched to {chosen_name}")
                    st.rerun()
                else:
                    st.sidebar.error(f"Switch tenant failed: {r.status_code} {r.text}")
    else:
        st.sidebar.caption("No memberships loaded. If needed, log out and log back in to refresh.")

# ---------- UI: Header ----------
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

# ---------- UI: Ingest ----------
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

# ---------- UI: Chat ----------
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

    # Render history (local echo only; Redis is the source of truth on the server)
    for m in st.session_state["messages"]:
        role = m.get("role", "user")
        with st.chat_message(role):
            st.write(m.get("text", ""))

    prompt = st.chat_input("Ask a question")
    if prompt:
        st.session_state["messages"].append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        body = {
            "session_id": "demo-session",
            "question": prompt,
            "k": 4,
            "provider": "stub",   # change once Gemini/Groq wired
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
                elif r.status_code == 401:
                    st.error("Unauthorized — your token might be missing or expired.")
                else:
                    st.error(f"Query failed: {r.status_code} {r.text}")
            except Exception as e:
                st.exception(e)

# ---------- Main ----------
def main():
    st.set_page_config(page_title="Smart RAG Chatbot", page_icon="🤖", layout="wide")

    # Sidebar tenant controls (post-login)
    tenant_sidebar()

    # Top header
    header_bar()

    jwt = _ss_get("jwt")
    tenant_id = _ss_get("tenant_id")

    # If not authenticated, show the auth panel only
    if not jwt:
        auth_panel()
        return

    # Authenticated: show app sections
    tab = st.sidebar.radio("Navigate", ["Chat", "Ingest", "Account"])
    if tab == "Chat":
        chat_panel()
    elif tab == "Ingest":
        ingest_panel()
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

# # streamlit_app.py
# import os
# import json
# import requests
# from typing import Dict, List, Optional, Any
# import streamlit as st

# # ---------- Config ----------
# API_BASE = os.getenv("API_BASE", "http://localhost:8000")
# TIMEOUT = 30

# # ---------- Session Helpers ----------
# def _ss_get(key: str, default=None):
#     if key not in st.session_state:
#         st.session_state[key] = default
#     return st.session_state[key]

# def _ss_set(**kwargs):
#     for k, v in kwargs.items():
#         st.session_state[k] = v

# def clear_auth_state():
#     for k in ("jwt", "tenant_id", "role", "email", "memberships"):
#         st.session_state.pop(k, None)

# def ensure_message_buffer():
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []  # list of {"role":"user|assistant", "text":"..."}

# # ---------- API Helpers ----------
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
#         # multipart/form-data (for file upload) or form fields
#         return requests.post(url, headers=headers, files=files, data=data, timeout=TIMEOUT)

# # ---------- UI: Auth ----------
# def auth_panel():
#     st.header("🔐 Sign in / Sign up")
#     tabs = st.tabs(["Login", "Sign up"])

#     # ---- Login tab (no tenant selection here) ----
#     with tabs[0]:
#         with st.form("login_form"):
#             email = st.text_input("Email", key="login_email")
#             password = st.text_input("Password", type="password", key="login_password")
#             submitted = st.form_submit_button("Login")
#         if submitted:
#             if not email or not password:
#                 st.error("Email and password are required.")
#             else:
#                 try:
#                     payload = {"email": email, "password": password}
#                     r = api_post("/auth/login", json_body=payload)
#                     if r.status_code == 200:
#                         data = r.json()
#                         # Possible shapes from your backend:
#                         # A) tenant token directly (rare for login now)
#                         if "token" in data:
#                             _ss_set(jwt=data["token"], tenant_id=data.get("tenant_id"), role=data.get("role"), email=email)
#                             st.success("Logged in.")
#                         # B) login token + memberships (tenant selection will be in sidebar)
#                         elif "login_token" in data:
#                             _ss_set(
#                                 jwt=data["login_token"],
#                                 tenant_id=None,
#                                 role=None,
#                                 email=email,
#                                 memberships=data.get("memberships", []),
#                             )
#                             st.info("Logged in. Pick a tenant from the sidebar to continue.")
#                         else:
#                             st.warning("Unexpected response shape.")
#                     else:
#                         st.error(f"Login failed: {r.status_code} {r.text}")
#                 except Exception as e:
#                     st.exception(e)

#     # ---- Sign up tab (kept with create/join flows) ----
#     with tabs[1]:
#         with st.form("signup_form"):
#             s_email = st.text_input("Email", key="signup_email")
#             s_password = st.text_input("Password", type="password", key="signup_password")

#             mode = st.radio(
#                 "Choose onboarding mode",
#                 ["Create a new tenant", "Join an existing tenant"],
#                 key="signup_mode"
#             )
#             tenant_name = ""
#             tenant_join = ""
#             if mode == "Create a new tenant":
#                 tenant_name = st.text_input("New tenant name", key="signup_tenant_name")
#             else:
#                 st.write("Search or paste a tenant slug/ID. (Enumeration allowed)")
#                 q = st.text_input("Search tenants", key="tenant_search_q")
#                 tenant_list = []
#                 if q.strip():
#                     resp = api_get("/auth/tenants/search", params={"q": q.strip()})
#                     if resp.status_code == 200:
#                         tenant_list = resp.json().get("tenants", [])
#                     else:
#                         st.warning(f"Search failed: {resp.status_code} {resp.text}")
#                 label_to_id = {f'{t["name"]} [{t["slug"]}]': t["id"] for t in tenant_list}
#                 default_label = next(iter(label_to_id.keys()), None)
#                 tenant_join_label = st.selectbox(
#                     "Choose a tenant",
#                     options=list(label_to_id.keys()) or ["(no results)"],
#                     index=0 if default_label else 0,
#                     key="tenant_join_pick"
#                 )
#                 if label_to_id:
#                     tenant_join = label_to_id.get(tenant_join_label, "")

#             submitted = st.form_submit_button("Create account")
#         if submitted:
#             if not s_email or not s_password:
#                 st.error("Email and password are required.")
#             else:
#                 body = {
#                     "email": s_email,
#                     "password": s_password,
#                 }
#                 if mode == "Create a new tenant":
#                     if not tenant_name.strip():
#                         st.error("Please provide a tenant name.")
#                     else:
#                         body.update({"mode": "create_tenant", "tenant_name": tenant_name.strip()})
#                 else:
#                     if not tenant_join:
#                         st.error("Select a tenant to join.")
#                     else:
#                         body.update({"mode": "join_tenant", "tenant_id": tenant_join})

#                 r = api_post("/auth/signup", json_body=body)
#                 if r.status_code == 200:
#                     data = r.json()
#                     # Sign-up issues tenant-scoped token directly in this model
#                     _ss_set(jwt=data["token"], tenant_id=data.get("tenant_id"), role=data.get("role"), email=s_email)
#                     st.success("Account created and tenant selected.")
#                 else:
#                     st.error(f"Signup failed: {r.status_code} {r.text}")

# # ---------- Sidebar: Tenant selection AFTER login ----------
# def tenant_sidebar():
#     jwt = _ss_get("jwt")
#     if not jwt:
#         st.sidebar.info("Log in to pick a tenant.")
#         return

#     st.sidebar.markdown("### 🏷️ Tenant")
#     current_tenant = _ss_get("tenant_id")
#     memberships = _ss_get("memberships", [])

#     if current_tenant:
#         st.sidebar.success(f"Current tenant: {current_tenant}")
#     else:
#         st.sidebar.warning("No tenant selected.")

#     # If we have memberships from login, offer a picker to switch
#     if memberships:
#         opts = {f'{m["tenant_name"]} ({m.get("role","")}) [{m.get("tenant_id")}]': m["tenant_id"] for m in memberships}
#         if opts:
#             choice = st.sidebar.selectbox("Choose a tenant to switch", list(opts.keys()), key="tenant_pick_sidebar")
#             if st.sidebar.button("Switch Tenant"):
#                 chosen_tenant_id = opts[choice]
#                 r = api_post("/auth/switch-tenant", token=jwt, json_body={"tenant_id": chosen_tenant_id})
#                 if r.status_code == 200:
#                     data = r.json()
#                     # Now we should have a tenant-scoped token
#                     _ss_set(jwt=data["token"], tenant_id=data["tenant_id"], role=data.get("role"))
#                     # cleanup memberships since we're now scoped
#                     st.session_state.pop("memberships", None)
#                     st.sidebar.success(f"Switched to tenant {data['tenant_id']}")
#                     st.rerun()
#                 else:
#                     st.sidebar.error(f"Switch tenant failed: {r.status_code} {r.text}")
#     else:
#         st.sidebar.caption("No memberships loaded. If needed, log out and log back in.")

# # ---------- UI: Header ----------
# def header_bar():
#     jwt = _ss_get("jwt")
#     tenant_id = _ss_get("tenant_id")
#     role = _ss_get("role")
#     email = _ss_get("email")
#     left, right = st.columns([0.7, 0.3])
#     with left:
#         st.caption("Smart RAG Chatbot — Demo UI")
#     with right:
#         if jwt:
#             if tenant_id:
#                 st.success(f"Tenant: {tenant_id}  |  Role: {role or '-'}")
#             else:
#                 st.warning("Tenant: (not selected)")
#             st.caption(f"Signed in as {email or '-'}")
#             if st.button("Log out"):
#                 clear_auth_state()
#                 st.rerun()
#         else:
#             st.warning("Not authenticated")

# # ---------- UI: Ingest ----------
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

# # ---------- UI: Chat ----------
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

#     # Render history (local echo only; Redis is the source of truth on the server)
#     for m in st.session_state["messages"]:
#         role = m.get("role", "user")
#         with st.chat_message(role):
#             st.write(m.get("text", ""))

#     prompt = st.chat_input("Ask a question")
#     if prompt:
#         st.session_state["messages"].append({"role": "user", "text": prompt})
#         with st.chat_message("user"):
#             st.write(prompt)

#         body = {
#             "session_id": "demo-session",  # or per-browser unique
#             "question": prompt,
#             "k": 4,
#             "provider": "stub",   # change once Gemini/Groq wired
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
#                 elif r.status_code == 401:
#                     st.error("Unauthorized — your token might be missing or expired.")
#                 else:
#                     st.error(f"Query failed: {r.status_code} {r.text}")
#             except Exception as e:
#                 st.exception(e)

# # ---------- Main ----------
# def main():
#     st.set_page_config(page_title="Smart RAG Chatbot", page_icon="🤖", layout="wide")

#     # Sidebar tenant controls (post-login)
#     tenant_sidebar()

#     # Top header
#     header_bar()

#     jwt = _ss_get("jwt")
#     tenant_id = _ss_get("tenant_id")

#     # If not authenticated, show the auth panel only
#     if not jwt:
#         auth_panel()
#         return

#     # Authenticated: show app sections
#     tab = st.sidebar.radio("Navigate", ["Chat", "Ingest", "Account"])
#     if tab == "Chat":
#         chat_panel()
#     elif tab == "Ingest":
#         ingest_panel()
#     else:
#         st.header("👤 Account")
#         st.write(f"Email: {_ss_get('email') or '-'}")
#         st.write(f"Tenant: {tenant_id or '(not selected)'}")
#         st.write(f"Role: {_ss_get('role') or '-'}")
#         if st.button("Log out (clear token)"):
#             clear_auth_state()
#             st.rerun()

# if __name__ == "__main__":
#     main()

# # # streamlit_app.py
# # import os
# # import time
# # import json
# # import requests
# # from typing import Dict, List, Optional, Any
# # import streamlit as st

# # # ---------- Config ----------
# # API_BASE = os.getenv("API_BASE", "http://localhost:8000")
# # TIMEOUT = 30

# # # ---------- Session Helpers ----------
# # def _ss_get(key: str, default=None):
# #     if key not in st.session_state:
# #         st.session_state[key] = default
# #     return st.session_state[key]

# # def _ss_set(**kwargs):
# #     for k, v in kwargs.items():
# #         st.session_state[k] = v

# # def clear_auth_state():
# #     for k in ("jwt", "tenant_id", "role", "email", "memberships"):
# #         st.session_state.pop(k, None)

# # def ensure_message_buffer():
# #     if "messages" not in st.session_state:
# #         st.session_state["messages"] = []  # list of {"role":"user|assistant", "text":"..."}

# # # ---------- API Helpers ----------
# # def api_get(path: str, token: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
# #     url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
# #     headers = {}
# #     if token:
# #         headers["Authorization"] = f"Bearer {token}"
# #     r = requests.get(url, headers=headers, params=params or {}, timeout=TIMEOUT)
# #     return r

# # def api_post(path: str, token: Optional[str] = None, json_body: Optional[Dict[str, Any]] = None, files=None, data=None):
# #     url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
# #     headers = {}
# #     if token:
# #         headers["Authorization"] = f"Bearer {token}"
# #     if files is None and data is None:
# #         headers["Content-Type"] = "application/json"
# #         r = requests.post(url, headers=headers, json=json_body or {}, timeout=TIMEOUT)
# #     else:
# #         # multipart/form-data (for file upload) or form fields
# #         r = requests.post(url, headers=headers, files=files, data=data, timeout=TIMEOUT)
# #     return r

# # # ---------- UI: Auth ----------
# # def auth_panel():
# #     st.header("🔐 Sign in / Sign up")

# #     tabs = st.tabs(["Login", "Sign up"])

# #     # ---- Login tab ----
# #     with tabs[0]:
# #         with st.form("login_form"):
# #             email = st.text_input("Email", key="login_email")
# #             password = st.text_input("Password", type="password", key="login_password")
# #             tenant_choice = st.text_input(
# #                 "Tenant (optional: paste tenant_id or slug to directly login to a tenant)",
# #                 key="login_tenant_opt",
# #                 help="If provided and you are a member, the server will issue a tenant-scoped token right away."
# #             )
# #             submitted = st.form_submit_button("Login")
# #         if submitted:
# #             if not email or not password:
# #                 st.error("Email and password are required.")
# #             else:
# #                 try:
# #                     payload = {"email": email, "password": password}
# #                     if tenant_choice.strip():
# #                         payload["tenant_id"] = tenant_choice.strip()
# #                     r = api_post("/auth/login", json_body=payload)
# #                     if r.status_code == 200:
# #                         data = r.json()
# #                         # Two possible shapes:
# #                         # A) tenant token directly
# #                         if "token" in data:
# #                             _ss_set(jwt=data["token"], tenant_id=data.get("tenant_id"), role=data.get("role"), email=email)
# #                             st.success("Logged in with tenant.")
# #                         # B) login token + memberships (choose tenant)
# #                         elif "login_token" in data and "memberships" in data:
# #                             _ss_set(jwt=data["login_token"], tenant_id=None, role=None, email=email, memberships=data["memberships"])
# #                             st.info("Choose a tenant below and press Switch Tenant.")
# #                         else:
# #                             st.warning("Unexpected response shape.")
# #                     else:
# #                         st.error(f"Login failed: {r.status_code} {r.text}")
# #                 except Exception as e:
# #                     st.exception(e)

# #         # If we got memberships back, show a selector to get tenant-scoped JWT
# #         memberships = _ss_get("memberships", [])
# #         if memberships:
# #             st.subheader("Pick a tenant")
# #             opts = {f'{m["tenant_name"]} ({m.get("role","")}) [{m.get("tenant_id")}]': m["tenant_id"] for m in memberships}
# #             choice = st.selectbox("Tenant", list(opts.keys()), key="tenant_pick")
# #             if st.button("Switch Tenant"):
# #                 chosen_tenant_id = opts[choice]
# #                 r = api_post("/auth/switch-tenant", token=_ss_get("jwt"), json_body={"tenant_id": chosen_tenant_id})
# #                 if r.status_code == 200:
# #                     data = r.json()
# #                     # Now we should have a tenant token
# #                     _ss_set(jwt=data["token"], tenant_id=data["tenant_id"], role=data.get("role"))
# #                     st.success(f"Switched to tenant {data['tenant_id']}")
# #                     # cleanup memberships list
# #                     st.session_state.pop("memberships", None)
# #                 else:
# #                     st.error(f"Switch tenant failed: {r.status_code} {r.text}")

# #     # ---- Sign up tab ----
# #     with tabs[1]:
# #         with st.form("signup_form"):
# #             s_email = st.text_input("Email", key="signup_email")
# #             s_password = st.text_input("Password", type="password", key="signup_password")

# #             mode = st.radio(
# #                 "Choose onboarding mode",
# #                 ["Create a new tenant", "Join an existing tenant"],
# #                 key="signup_mode"
# #             )
# #             tenant_name = ""
# #             tenant_join = ""
# #             if mode == "Create a new tenant":
# #                 tenant_name = st.text_input("New tenant name", key="signup_tenant_name")
# #             else:
# #                 st.write("Search or paste a tenant slug/ID. (Enumeration allowed)")
# #                 q = st.text_input("Search tenants", key="tenant_search_q")
# #                 tenant_list = []
# #                 if q.strip():
# #                     resp = api_get("/auth/tenants/search", params={"q": q.strip()})
# #                     if resp.status_code == 200:
# #                         tenant_list = resp.json().get("tenants", [])
# #                     else:
# #                         st.warning(f"Search failed: {resp.status_code} {resp.text}")
# #                 label_to_id = {f'{t["name"]} [{t["slug"]}]': t["id"] for t in tenant_list}
# #                 default_label = next(iter(label_to_id.keys()), None)
# #                 tenant_join_label = st.selectbox("Choose a tenant", options=list(label_to_id.keys()) or ["(no results)"], index=0 if default_label else 0, key="tenant_join_pick")
# #                 if label_to_id:
# #                     tenant_join = label_to_id.get(tenant_join_label, "")

# #             submitted = st.form_submit_button("Create account")
# #         if submitted:
# #             if not s_email or not s_password:
# #                 st.error("Email and password are required.")
# #             else:
# #                 body = {
# #                     "email": s_email,
# #                     "password": s_password,
# #                 }
# #                 if mode == "Create a new tenant":
# #                     if not tenant_name.strip():
# #                         st.error("Please provide a tenant name.")
# #                     else:
# #                         body.update({"mode": "create_tenant", "tenant_name": tenant_name.strip()})
# #                 else:
# #                     if not tenant_join:
# #                         st.error("Select a tenant to join.")
# #                     else:
# #                         body.update({"mode": "join_tenant", "tenant_id": tenant_join})

# #                 r = api_post("/auth/signup", json_body=body)
# #                 if r.status_code == 200:
# #                     data = r.json()
# #                     # Sign-up issues tenant-scoped token directly in this simple model
# #                     _ss_set(jwt=data["token"], tenant_id=data.get("tenant_id"), role=data.get("role"), email=s_email)
# #                     st.success("Account created and tenant selected.")
# #                 else:
# #                     st.error(f"Signup failed: {r.status_code} {r.text}")

# # # ---------- UI: Header ----------
# # def header_bar():
# #     jwt = _ss_get("jwt")
# #     tenant_id = _ss_get("tenant_id")
# #     role = _ss_get("role")
# #     email = _ss_get("email")
# #     left, right = st.columns([0.7, 0.3])
# #     with left:
# #         st.caption("Smart RAG Chatbot — Demo UI")
# #     with right:
# #         if jwt and tenant_id:
# #             st.success(f"Tenant: {tenant_id}  |  Role: {role or '-'}")
# #             st.caption(f"Signed in as {email or '-'}")
# #             if st.button("Log out"):
# #                 clear_auth_state()
# #                 st.experimental_rerun()
# #         else:
# #             st.warning("Not authenticated")

# # # ---------- UI: Ingest ----------
# # def ingest_panel():
# #     st.header("📥 Ingest")
# #     jwt = _ss_get("jwt")
# #     tenant_id = _ss_get("tenant_id")

# #     if not jwt or not tenant_id:
# #         st.info("Please log in and select a tenant first.")
# #         return

# #     st.subheader("Upload a file")
# #     f = st.file_uploader("PDF or TXT", type=["pdf", "txt"], key="upload_file")
# #     if st.button("Ingest File"):
# #         if f is None:
# #             st.error("Choose a file first.")
# #         else:
# #             files = {"file": (f.name, f.getvalue(), f"type" if hasattr(f, "type") else "application/octet-stream")}
# #             r = api_post("/ingest", token=jwt, files=files)
# #             if r.status_code == 200:
# #                 st.success(f"Ingested: {r.json()}")
# #             else:
# #                 st.error(f"Upload failed: {r.status_code} {r.text}")

# #     st.subheader("Ingest URLs")
# #     urls_csv = st.text_area("Enter one or more URLs (comma or newline separated)")
# #     if st.button("Ingest URLs"):
# #         urls = [u.strip() for u in urls_csv.replace("\n", ",").split(",") if u.strip()]
# #         if not urls:
# #             st.error("Provide at least one URL.")
# #         else:
# #             r = api_post("/ingest", token=jwt, json_body={"urls": urls})
# #             if r.status_code == 200:
# #                 st.success(f"Ingested: {r.json()}")
# #             else:
# #                 st.error(f"URL ingest failed: {r.status_code} {r.text}")

# # # ---------- UI: Chat ----------
# # def chat_panel():
# #     st.header("💬 Chat")
# #     jwt = _ss_get("jwt")
# #     tenant_id = _ss_get("tenant_id")
# #     if not jwt or not tenant_id:
# #         st.info("Please log in and select a tenant first.")
# #         return

# #     ensure_message_buffer()

# #     # Render history (local echo only; Redis is the source of truth on the server)
# #     for m in st.session_state["messages"]:
# #         role = m.get("role", "user")
# #         with st.chat_message(role):
# #             st.write(m.get("text", ""))

# #     # Input
# #     prompt = st.chat_input("Ask a question")
# #     if prompt:
# #         st.session_state["messages"].append({"role": "user", "text": prompt})
# #         with st.chat_message("user"):
# #             st.write(prompt)

# #         # Call backend /query
# #         body = {
# #             "session_id": "demo-session",  # or something unique per browser/user
# #             "question": prompt,
# #             "k": 4,
# #             "provider": "stub",      # you can change once you wire Gemini/Groq
# #             "use_mmr": True,
# #             "mmr_lambda": 0.7,
# #         }
# #         with st.chat_message("assistant"):
# #             try:
# #                 r = api_post("/query", token=jwt, json_body=body)
# #                 if r.status_code == 200:
# #                     data = r.json()
# #                     answer = data.get("answer", "")
# #                     sources = data.get("sources", [])
# #                     st.write(answer)
# #                     if sources:
# #                         st.caption("Sources: " + ", ".join(sources))
# #                     st.session_state["messages"].append({"role": "assistant", "text": answer})
# #                 elif r.status_code == 401:
# #                     st.error("Unauthorized — your token might be missing or expired.")
# #                 else:
# #                     st.error(f"Query failed: {r.status_code} {r.text}")
# #             except Exception as e:
# #                 st.exception(e)

# # # ---------- Main ----------
# # def main():
# #     st.set_page_config(page_title="Smart RAG Chatbot", page_icon="🤖", layout="wide")
# #     header_bar()

# #     jwt = _ss_get("jwt")
# #     tenant_id = _ss_get("tenant_id")

# #     # If not authenticated, show the auth panel only
# #     if not jwt or not tenant_id:
# #         auth_panel()
# #         return

# #     # Authenticated: show app sections
# #     tab = st.sidebar.radio("Navigate", ["Chat", "Ingest", "Account"])
# #     if tab == "Chat":
# #         chat_panel()
# #     elif tab == "Ingest":
# #         ingest_panel()
# #     else:
# #         st.header("👤 Account")
# #         st.write(f"Email: {_ss_get('email') or '-'}")
# #         st.write(f"Tenant: {tenant_id}")
# #         st.write(f"Role: {_ss_get('role') or '-'}")
# #         if st.button("Log out (clear token)"):
# #             clear_auth_state()
# #             st.experimental_rerun()

# # if __name__ == "__main__":
# #     main()

# # # # ui/streamlit_app.py  (put anywhere; I like /ui)
# # # from __future__ import annotations

# # # import os
# # # import uuid
# # # import time
# # # import requests
# # # from typing import Dict, Any, List, Optional

# # # import streamlit as st


# # # # ---------- Config ----------
# # # DEFAULT_API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# # # st.set_page_config(page_title="Smart RAG Chatbot", page_icon="🤖", layout="wide")

# # # if "session_id" not in st.session_state:
# # #     st.session_state.session_id = str(uuid.uuid4())

# # # if "messages" not in st.session_state:
# # #     # local echo of the conversation for display only; Redis keeps the “truth”
# # #     st.session_state.messages = []


# # # # ---------- Helpers ----------
# # # def auth_headers(token: Optional[str]) -> Dict[str, str]:
# # #     return {"Authorization": f"Bearer {token}"} if token else {}


# # # def api_post(path: str, json: Optional[dict] = None, files=None, data=None, token: Optional[str] = None):
# # #     url = f"{st.session_state.api_base}{path}"
# # #     headers = auth_headers(token)
# # #     return requests.post(url, json=json, files=files, data=data, headers=headers, timeout=60)


# # # def api_get(path: str, token: Optional[str] = None):
# # #     url = f"{st.session_state.api_base}{path}"
# # #     headers = auth_headers(token)
# # #     return requests.get(url, headers=headers, timeout=20)


# # # def render_sources(sources: List[str] | List[dict]) -> str:
# # #     if not sources:
# # #         return ""
# # #     # our API returns a simple list[str] today; B2 metadata also has {source, chunk_id}
# # #     if isinstance(sources[0], dict):
# # #         parts = [f"{s.get('source','?')}#{s.get('chunk_id','')}" for s in sources]  # type: ignore[index]
# # #     else:
# # #         parts = [str(s) for s in sources]  # type: ignore[assignment]
# # #     return "Sources: " + ", ".join(parts)


# # # # ---------- Sidebar ----------
# # # with st.sidebar:
# # #     st.markdown("## Settings")

# # #     api_default = st.session_state.get("api_base", DEFAULT_API_BASE)
# # #     st.session_state.api_base = st.text_input("API base", api_default, help="FastAPI base URL")

# # #     st.session_state.jwt = st.text_input(
# # #         "JWT Bearer token",
# # #         value=st.session_state.get("jwt", ""),
# # #         type="password",
# # #         help="Required for /ingest and /query",
# # #     )

# # #     provider = st.selectbox(
# # #         "Provider",
# # #         options=["gemini",  "groq", "stub"],
# # #         index=["gemini",  "groq", "stub"].index("gemini"),
# # #     )

# # #     mmr_used = st.toggle("Use MMR (diversity)", value=False)
# # #     mmr_lambda = st.slider("MMR λ (relevance↔diversity)", 0.0, 1.0, 0.5, 0.05, disabled=not mmr_used)
# # #     top_k = st.slider("Top-K", 1, 10, 4, 1)

# # #     st.divider()
# # #     st.markdown("### Rebuild / Ingest")

# # #     upl = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
# # #     urls_text = st.text_area("URLs (comma/newline separated)")

# # #     col_ing_a, col_ing_b = st.columns(2)
# # #     with col_ing_a:
# # #         if st.button("Ingest", use_container_width=True):
# # #             try:
# # #                 with st.spinner("Ingesting…"):
# # #                     # Prepare multipart if file present
# # #                     files = None
# # #                     data = {}
# # #                     if upl is not None:
# # #                         files = {"file": (upl.name, upl, upl.type or "application/octet-stream")}
# # #                     # URLs can be sent as JSON OR as form; we’ll prefer JSON call if no file,
# # #                     # otherwise send form field 'urls' so backend path works for multipart.
# # #                     if urls_text.strip():
# # #                         urls = [u.strip() for u in urls_text.replace(",", "\n").splitlines() if u.strip()]
# # #                     else:
# # #                         urls = []

# # #                     if files and urls:
# # #                         # multipart with form field 'urls' (can repeat)
# # #                         # requests supports repeated keys via list of tuples:
# # #                         data = [("urls_form", ",".join(urls))]
# # #                         resp = api_post("/ingest", files=files, data=data, token=st.session_state.jwt)
# # #                     elif files:
# # #                         resp = api_post("/ingest", files=files, token=st.session_state.jwt)
# # #                     else:
# # #                         resp = api_post("/ingest", json={"urls": urls}, token=st.session_state.jwt)

# # #                     if resp.status_code == 200:
# # #                         js = resp.json()
# # #                         st.success(
# # #                             f"Ingested {js.get('ingested', 0)} chunks "
# # #                             f"(load={js['durations_ms'].get('load')}ms, index={js['durations_ms'].get('index')}ms)"
# # #                         )
# # #                         if js.get("failures"):
# # #                             st.warning("Failures: " + "; ".join(js["failures"]))
# # #                     elif resp.status_code == 401:
# # #                         st.error("Unauthorized (401). Check your JWT token in the sidebar.")
# # #                     else:
# # #                         st.error(f"Error {resp.status_code}: {resp.text}")
# # #             except Exception as e:
# # #                 st.exception(e)
# # #     with col_ing_b:
# # #         if st.button("Health check", use_container_width=True):
# # #             try:
# # #                 r = api_get("/health")
# # #                 if r.ok:
# # #                     st.success(r.json())
# # #                 else:
# # #                     st.error(f"{r.status_code}: {r.text}")
# # #             except Exception as e:
# # #                 st.exception(e)

# # #     st.divider()
# # #     st.caption(f"Session: `{st.session_state.session_id}`")


# # # # ---------- Main Chat ----------
# # # st.title("Smart RAG Chatbot by Prashanth Madishetti")

# # # # show existing conversation
# # # for m in st.session_state.messages:
# # #     with st.chat_message(m["role"]):
# # #         st.markdown(m["content"])

# # # # chat input
# # # prompt = st.chat_input("Ask me something…")
# # # if prompt:
# # #     # display immediately
# # #     st.session_state.messages.append({"role": "user", "content": prompt})
# # #     with st.chat_message("user"):
# # #         st.markdown(prompt)

# # #     # call backend
# # #     try:
# # #         t0 = time.time()
# # #         body = {
# # #             "question": prompt,
# # #             "session_id": st.session_state.session_id,
# # #             "k": int(top_k),
# # #             "provider": provider,
# # #         }
# # #         # optional B1 controls if your /query supports them
# # #         if mmr_used:
# # #             body["use_mmr"] = True
# # #             body["mmr_lambda"] = float(mmr_lambda)

# # #         resp = api_post("/query", json=body, token=st.session_state.jwt)
# # #         t1 = time.time()
# # #         latency_ms = int((t1 - t0) * 1000)

# # #         if resp.status_code == 200:
# # #             js = resp.json()
# # #             answer = js.get("answer", "")
# # #             provider_used = js.get("provider", provider)
# # #             sources = js.get("sources", [])

# # #             with st.chat_message("assistant"):
# # #                 st.markdown(answer)
# # #                 meta_line = f"_{provider_used}_ • {render_sources(sources)} • {latency_ms}ms"
# # #                 st.caption(meta_line)

# # #             st.session_state.messages.append(
# # #                 {"role": "assistant", "content": f"{answer}\n\n_{provider_used}_"}
# # #             )
# # #         elif resp.status_code == 401:
# # #             with st.chat_message("assistant"):
# # #                 st.error("Unauthorized (401). Add a valid JWT token in the sidebar.")
# # #         else:
# # #             with st.chat_message("assistant"):
# # #                 st.error(f"Error {resp.status_code}: {resp.text}")

# # #     except Exception as e:
# # #         with st.chat_message("assistant"):
# # #             st.exception(e)