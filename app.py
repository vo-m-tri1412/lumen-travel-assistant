"""Lumen Travel: signed-in, source-grounded travel chat for Ho Chi Minh City."""

import streamlit as st

from config import ROOT, Settings
from firebase_client import AuthError, FirebaseClient, clear_session
from rag import GuideIndex, OllamaClient, RagError, answer_question, guide_fingerprint, load_guide

st.set_page_config(page_title="Lumen Travel · Sài Gòn", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
st.html("""<style>
.stApp{background:radial-gradient(ellipse at 90% 0%,#e9e1f8 0,transparent 42%),#fcfaf7}
.block-container{max-width:1140px;padding-top:2.6rem;padding-bottom:2rem}
[data-testid="stSidebar"]{background:#f1eee9;border-right:1px solid #e6e0d7}
h1,h2,h3{letter-spacing:-.035em}h1{font-weight:650!important}
[data-testid="stForm"]{border:1px solid #e8e1d9;border-radius:24px;padding:25px;background:#ffffffb8}
[data-testid="stChatMessage"]{border-radius:20px;border:1px solid #ebe5df;background:#fffdfb;padding:20px;margin-bottom:14px}
[data-testid="stChatInput"]{border-radius:19px}
.lumen-brand{font-size:29px;font-weight:750;letter-spacing:-1.2px;color:#29283a}
.lumen-star{color:#7853c8;margin-right:9px}.city-label{font-size:12px;letter-spacing:.14em;color:#8b7d9b;margin-top:3px}
.welcome{margin:20px 0 25px;padding:34px;border-radius:26px;background:linear-gradient(125deg,#e1f0ea,#eee3f8)}
.welcome h1{font-size:43px;margin:0 0 12px}.welcome p{color:#695e77;margin:0;font-size:17px}
.guide-pill{display:inline-block;border:1px solid #cbbfdb;border-radius:30px;padding:5px 13px;margin:17px 7px 0 0;color:#695586;font-size:13px}
.auth-visual{padding:55px 34px 20px}.auth-visual h1{font-size:52px;line-height:1.12;margin:15px 0 24px}
.source-title{font-weight:650;color:#674b9a;margin-top:9px}.source-score{font-size:12px;color:#887b95}
button[kind="primary"]{border-radius:12px}.stButton button{border-radius:12px}
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
<p>Hỏi điều bạn muốn biết, xem nguồn cho từng câu trả lời.</p>
<span class="guide-pill">Bảo tàng</span><span class="guide-pill">Dạo phố</span><span class="guide-pill">Văn hóa</span>
</div>''')
    with right:
        st.subheader("Bắt đầu hành trình")
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
        st.caption("ĐANG ĐĂNG NHẬP")
        st.write(session.email)
        st.caption("16 mục trong cẩm nang")
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
