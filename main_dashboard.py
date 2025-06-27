import streamlit as st
import os
import sys
import types
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import json
import time

# ─── Page Config ───────────────────────────────
st.set_page_config(page_title="Growify AI Toolkit", layout="wide")

# ─── THEME (if you have a theme.py) ─────────────
try:
    from theme import apply_dark_theme
    apply_dark_theme()
except ImportError:
    pass

# ─── Firebase Config ───────────────────────────
firebase_config = {
    "apiKey":            os.getenv("FIREBASE_API_KEY"),
    "authDomain":        os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL":       os.getenv("FIREBASE_DATABASE_URL"),
    "projectId":         os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket":     os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId":             os.getenv("FIREBASE_APP_ID"),
}

try:
    pb = pyrebase.initialize_app(firebase_config)
    firebase_auth = pb.auth()
except Exception as e:
    st.error(f"❌ Firebase init error: {e}")
    st.stop()

if not firebase_admin._apps:
    service_account_json = os.getenv("FIREBASE_ADMIN_JSON")
    if not service_account_json:
        st.error("❌ FIREBASE_ADMIN_JSON environment variable is not set!")
        st.stop()
    try:
        cred = credentials.Certificate(json.loads(service_account_json))
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"❌ Failed to initialize Firebase Admin SDK: {e}")
        st.stop()

db = firestore.client()
# ─── Firebase Helpers ──────────────────────────
def log_user_login(user_id, email):
    user_ref = db.collection('user_activity').document(user_id)
    user_ref.set({
        'email': email,
        'login_time': firestore.SERVER_TIMESTAMP,
        'logged_in': True
    }, merge=True)
    st.session_state.login_time = time.time()

def log_user_logout(user_id):
    user_ref = db.collection('user_activity').document(user_id)
    logout_time = time.time()
    user_ref.update({
        'logout_time': firestore.SERVER_TIMESTAMP,
        'logged_in': False
    })
    session_duration = logout_time - st.session_state.login_time
    user_ref.update({'session_duration': session_duration})

# ─── Login / Logout UI ─────────────────────────
def do_login():
    st.title("Company Portal Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        pwd   = st.text_input("Password", type="password")
        submit = st.form_submit_button("Log In")
        if submit:
            try:
                user = firebase_auth.sign_in_with_email_and_password(email.strip(), pwd)
                if user:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    user_id = user['localId']
                    log_user_login(user_id, email)
                    st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Network error: {e}")
            except Exception as e:
                st.error(f"❌ Login failed. {e}")

def do_logout():
    with st.sidebar:
        if st.button("Log out"):
            user_id = st.session_state.user['localId']
            log_user_logout(user_id)
            st.session_state.clear()
            st.rerun()

# ─── Session Check ─────────────────────────────
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    do_login()
    st.stop()
else:
    do_logout()

# ─── Patch for torch.classes (rembg / ultralytics) ─
class DummyModule(types.ModuleType):
    __path__ = []
sys.modules["torch.classes"] = DummyModule("torch.classes")

# ─── Add Project Folders to sys.path ───────────
base_dir = os.path.dirname(__file__)
projects = {
    "blog": os.path.join(base_dir, "GF_BLOG_IDEA"),
    "img": os.path.join(base_dir, "GF_IMG_RE"),
    "seo": os.path.join(base_dir, "GF_SEO"),
    "scraper": os.path.join(base_dir, "GF_SCRAPERS"),
    "pixelmatch": os.path.join(base_dir, "GF_IMG_FIND"),
}
for path in projects.values():
    if path not in sys.path:
        sys.path.insert(0, path)

# ─── Import Tool Modules ───────────────────────
try:
    from GF_BLOG_IDEA import blog_ideas as blog_module
except ImportError:
    blog_module = None

try:
    from GF_IMG_RE import app as img_module
except ImportError as e:
    img_module = None
    print("🔥 Failed to import img_module:", e)

try:
    from GF_SEO import app as seo_module
except ImportError:
    seo_module = None

try:
    from GF_SCRAPERS import app as scraper_module
except ImportError as e:
    scraper_module = None
    print("🔥 Failed to import scraper_module:", e)

try:
    from GF_IMG_FIND import app as pixelmatch_module
except ImportError as e:
    pixelmatch_module = None
    print("🔥 Failed to import PixelMatch module:", e)

# ─── Session State Tracking ─────────────────────
if "tool" not in st.session_state:
    st.session_state.tool = "home"

if st.session_state.tool != "home":
    with st.sidebar:
        if st.button("Back to Home"):
            st.session_state.tool = "home"
            st.rerun()

# ─── Home Screen ────────────────────────────────
def render_home():
    st.title("🧰 Growify Master AI Toolkit")
    st.markdown("#### 👇 Select a tool to get started")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Promptly", use_container_width=True,
                  on_click=lambda: st.session_state.update(tool="blog"))
    
    with col2:
        st.button("Snipster", use_container_width=True,
                  on_click=lambda: st.session_state.update(tool="img"))
    
    with col3:
        st.button("MetaScan", use_container_width=True,
                  on_click=lambda: st.session_state.update(tool="seo"))

    col4, col5 = st.columns(2)
    with col4:
        st.button("PixelMatch", use_container_width=True,
                  on_click=lambda: st.session_state.update(tool="pixelmatch"))
    with col5:
        st.button("PitchKit", use_container_width=True,
                  on_click=lambda: st.session_state.update(tool="scraper"))

# ─── Route Based on Selection ───────────────────
if st.session_state.tool == "home":
    render_home()
elif st.session_state.tool == "blog":
    if blog_module: blog_module.main()
    else: st.error("⚠️ Blog Idea module not found.")
elif st.session_state.tool == "img":
    if img_module: img_module.main()
    else: st.error("⚠️ Image Resizer module not found.")
elif st.session_state.tool == "pixelmatch":
    if pixelmatch_module: pixelmatch_module.main()
    else: st.error("⚠️ PixelMatch module not found.")
elif st.session_state.tool == "seo":
    if seo_module: seo_module.main()
    else: st.error("⚠️ SEO module not found.")
elif st.session_state.tool == "scraper":
    if scraper_module: scraper_module.main()
    else: st.error("⚠️ Scraper module not found.")
