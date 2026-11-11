"""Lumen Travel: signed-in, source-grounded travel chat for Ho Chi Minh City."""

import streamlit as st

from config import ROOT, Settings
from firebase_client import AuthError, FirebaseClient, clear_session
from rag import GuideIndex, OllamaClient, RagError, answer_question, guide_fingerprint, load_guide

st.set_page_config(page_title="Lumen Travel · Sài Gòn", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
st.html("""<style>
.stApp{background:radial-gradient(ellipse at 90% 0%,#f0e8ff 0,transparent 38%),#fcfbff}
.block-container{max-width:1200px;padding-top:2.2rem;padding-bottom:2rem}
[data-testid="stSidebar"]{background:#f7f3ff;border-right:1px solid #ece5fa;min-width:205px;max-width:235px}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]{overflow-wrap:anywhere;font-size:14px}
h1,h2,h3{letter-spacing:-.035em}h1{font-weight:650!important}
[data-testid="stForm"]{border:1px solid #e9e2f5;border-radius:25px;padding:27px;background:white;box-shadow:0 10px 35px #6c40b409}
[data-testid="stChatMessage"]{border-radius:23px;border:1px solid #ece7f4;background:white;padding:23px;margin-bottom:16px;box-shadow:0 6px 20px #7144ae05}
[data-testid="stChatInput"]{border-radius:20px;border-color:#dfd1fa;background:white}
[data-testid="stTextInput"] input{border-radius:11px}
[data-baseweb="tab-list"]{gap:25px}
.lumen-brand{font-size:30px;font-weight:750;letter-spacing:-1.2px;color:#292238}
.lumen-star{display:inline-grid;place-items:center;background:linear-gradient(145deg,#b77bff,#7e3ff2);color:white;width:38px;height:38px;border-radius:13px;margin-right:10px;font-size:24px;box-shadow:0 5px 18px #934af425}
.city-label{font-size:11px;letter-spacing:.15em;color:#8d7da5;margin:7px 0 0 50px}
.welcome{position:relative;margin:8px 0 26px;padding:42px 35px;border-radius:28px;background:radial-gradient(ellipse at 0 0,#a9e9ed,transparent 65%),radial-gradient(ellipse at 100% 0,#d6bbff,transparent 68%),linear-gradient(#f3edff,#ffffff);min-height:210px;text-align:center;border:1px solid #ece6fc}
.welcome:after{content:'✧';position:absolute;right:28px;top:18px;font-size:48px;color:#ab82e66b}
.welcome h1{font-size:44px;margin:14px 0 13px;background:linear-gradient(90deg,#2887ff,#8351ee);background-clip:text;-webkit-background-clip:text;color:transparent}
.welcome p{color:#75658d;margin:0;font-size:16px}
.guide-pill{display:inline-block;background:#ffffffa8;border:1px solid #d9c8f5;border-radius:30px;padding:7px 15px;margin:16px 6px 0 0;color:#7950b4;font-size:14px}
.auth-visual{position:relative;padding:51px 35px 43px;border-radius:29px;margin-top:28px;background:radial-gradient(ellipse at 0 0,#b1edf0,transparent 68%),radial-gradient(ellipse at 100% 0,#d5baff,transparent 69%),#faf8ff;border:1px solid #ede6fa;min-height:350px}
.auth-visual:after{content:'✧';position:absolute;right:23px;top:18px;font-size:49px;color:#b598de90}
.auth-visual h1{font-size:52px;line-height:1.15;margin:15px 0 24px;background:linear-gradient(110deg,#2a89f9,#9350eb);background-clip:text;-webkit-background-clip:text;color:transparent}
.source-title{font-weight:650;color:#674b9a;margin-top:9px}.source-score{font-size:12px;color:#887b95}
button[kind="primary"]{border-radius:13px;background:linear-gradient(105deg,#a353f2,#7352e8);border:0;box-shadow:0 5px 17px #8c4ae626}.stButton button{border-radius:13px;border-color:#e3d9f3}
[data-testid="stToolbar"]{visibility:hidden}
footer{visibility:hidden}
</style>""")


def brand():
    st.html('<div class="lumen-brand"><span class="lumen-star">✦</span>lumen</div><div class="city-label">TRAVEL · SÀI GÒN</div>')


@st.cache_resource(show_spinner=False)
def cached_index(base_url, token, fingerprint, entries):
    # The fingerprint changes when the guide changes. Public guide vectors are
    # shared between users; their questions and chat history are never cached.
    return GuideIndex.build(entries, OllamaClient(base_url, token))


def sources(hits):
    if not hits:
        return
    with st.expander(f"Nguồn đã tìm · {len(hits)} mục", expanded=False):
        for hit in hits:
            entry = hit["entry"]
            st.markdown(f'**[{entry["id"]}] {entry["title"]}** · cosine `{hit["score"]:.3f}`')
            st.write(entry["text"])
            st.markdown(f'[Đọc nguồn gốc]({entry["source"]})')


def login_page(settings):
    left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with left:
        brand()
        st.html('''<div class="auth-visual"><h1>Khám phá<br>Sài Gòn.</h1>
<span class="guide-pill">Bảo tàng</span><span class="guide-pill">Dạo phố</span><span class="guide-pill">Văn hóa</span>
</div>''')
    with right:
        st.subheader("Chào bạn")
        if not settings.firebase_api_key:
            st.info("Ứng dụng chưa được cấu hình đăng nhập. Xem README để thiết lập Firebase.")
        tabs = st.tabs(["Đăng nhập", "Tạo tài khoản"])
        for tab, signup in zip(tabs, [False, True]):
            with tab:
                prefix = "signup" if signup else "login"
                with st.form(prefix, clear_on_submit=True):
                    email = st.text_input("Email", placeholder="ban@example.com", key=prefix+"_email")
                    password = st.text_input("Mật khẩu", type="password", key=prefix+"_password",
                                             help="Ít nhất 6 ký tự." if signup else None)
                    confirmation = st.text_input("Nhập lại mật khẩu", type="password", key="signup_confirmation") if signup else None
                    submit = st.form_submit_button("Tạo tài khoản" if signup else "Đăng nhập",
                                                   type="primary", use_container_width=True,
                                                   disabled=not settings.firebase_api_key)
                if submit:
                    if signup and password != confirmation:
                        st.error("Hai mật khẩu chưa khớp nhau.")
                        continue
                    try:
                        session = FirebaseClient(settings.firebase_api_key).authenticate(email, password, signup)
                        clear_session(st.session_state)
                        st.session_state.auth = session
                        st.session_state.messages = []
                        st.rerun()
                    except AuthError as error:
                        st.error(str(error))


def chat_page(settings, session):
    categories = {"museum":"Bảo tàng", "market":"Chợ", "heritage":"Di sản", "architecture":"Kiến trúc",
                  "theatre":"Nhà hát", "street":"Phố", "park":"Công viên", "pagoda":"Chùa",
                  "walking_street":"Phố đi bộ", "viewpoint":"Ngắm cảnh", "river_transport":"Đường sông"}
    with st.sidebar:
        brand()
        st.divider()
        st.write(session.email)
        selected_category = st.selectbox("Chủ đề", [None, *categories],
                                         format_func=lambda value: categories.get(value, "Tất cả"))
        if st.button("Đăng xuất", use_container_width=True):
            clear_session(st.session_state)
            st.rerun()
    st.html('<div class="welcome"><h1>Bạn muốn khám phá đâu?</h1><p>TP. Hồ Chí Minh</p></div>')
    if not settings.ollama_base_url:
        st.info("Kết nối cẩm nang chưa được cấu hình. Xem README để thiết lập Ollama + Pinggy.")
        return
    messages = st.session_state.setdefault("messages", [])
    if not messages:
        cols = st.columns(3)
        for col, label in zip(cols, ["Giá vé Dinh Độc Lập?", "Bảo tàng về chiến tranh?", "Dạo phố Nguyễn Huệ?"]):
            if col.button(label, use_container_width=True):
                st.session_state.suggested_question = label
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources(message.get("hits", []))
    submitted = st.chat_input("Hỏi về một địa điểm trong cẩm nang…", max_chars=1200)
    question = st.session_state.pop("suggested_question", None) or submitted
    if question is None:
        return
    try:
        if not question.strip():
            raise RagError("Bạn nhập câu hỏi trước nhé.")
        # Validate with Firebase again before the protected operation.
        st.session_state.auth = FirebaseClient(settings.firebase_api_key).validate(st.session_state.auth)
        entries = load_guide(ROOT / "knowledge_base.json")
        with st.spinner("Đang tra cẩm nang…"):
            index = cached_index(settings.ollama_base_url, settings.ollama_api_token,
                                 guide_fingerprint(entries), entries)
            result = answer_question(question, index, OllamaClient(settings.ollama_base_url, settings.ollama_api_token),
                                     settings.top_k, settings.min_similarity, selected_category)
        messages.extend([{"role": "user", "content": question},
                         {"role": "assistant", "content": result["answer"], "hits": result["hits"]}])
        st.rerun()
    except AuthError as error:
        clear_session(st.session_state)
        st.session_state.auth_error = str(error)
        st.rerun()
    except (RagError, OSError, ValueError) as error:
        st.error(str(error))


def main():
    try:
        settings = Settings.load()
    except ValueError as error:
        st.error(str(error))
        st.stop()
    if "auth_error" in st.session_state:
        st.warning(st.session_state.pop("auth_error"))
    session = st.session_state.get("auth")
    if session:
        try:
            session = FirebaseClient(settings.firebase_api_key).validate(session)
            st.session_state.auth = session
        except AuthError as error:
            clear_session(st.session_state)
            st.warning(str(error))
            session = None
    if session:
        chat_page(settings, session)
    else:
        login_page(settings)


main()
