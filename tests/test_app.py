import os
import time
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from firebase_client import AuthError, AuthSession

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.dict(os.environ, {"FIREBASE_API_KEY":"test-key", "OLLAMA_BASE_URL":"https://example.org"})
        self.config.start()
        self.addCleanup(self.config.stop)

    def test_anonymous_user_cannot_see_chat(self):
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_input), 0)
        self.assertEqual(len(app.text_input), 5)

    @patch("firebase_client.FirebaseClient.authenticate", side_effect=AuthError("Email hoặc mật khẩu chưa đúng."))
    def test_failed_login_remains_outside_chat(self, authenticate):
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        app.text_input[0].set_value("test@example.com")
        app.text_input[1].set_value("incorrect")
        app.button[0].click().run()
        self.assertEqual(len(app.chat_input), 0)
        self.assertIn("chưa đúng", app.error[0].value)

    @patch("firebase_client.FirebaseClient.validate")
    @patch("firebase_client.FirebaseClient.authenticate")
    def test_signin_enables_chat_and_signout_removes_history(self, authenticate, validate):
        session = AuthSession("test@example.com", "u1", "test-id-token", "test-refresh", time.time()+3600)
        authenticate.return_value = session
        validate.return_value = session
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        app.text_input[0].set_value(session.email)
        app.text_input[1].set_value("test-password")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_input), 1)
        app.session_state["messages"] = [{"role":"user", "content":"A private question"}]
        next(button for button in app.button if button.label=="Đăng xuất").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_input), 0)
        self.assertNotIn("messages", app.session_state)
        self.assertNotIn("auth", app.session_state)

    @patch("firebase_client.FirebaseClient.validate")
    @patch("rag.requests.post")
    def test_two_questions_reuse_one_cached_guide_embedding(self, post, validate):
        from streamlit import cache_resource
        cache_resource.clear()
        session = AuthSession("test@example.com", "u1", "id", "refresh", time.time()+3600)
        validate.return_value = session

        def response(url, **kwargs):
            body=kwargs["json"]
            result=Mock(ok=True)
            if url.endswith("/api/embed"):
                count=len(body["input"])
                vectors=[[float(i==j) for j in range(16)] for i in range(count)]
                result.json.return_value={"embeddings":vectors}
            else:
                result.json.return_value={"message":{"content":json.dumps({"answerable":True,"sentences":[{"entry_id":"G01","quote":"Dinh Độc Lập ở 135 Nam Kỳ Khởi Nghĩa."}]})}}
            return result

        post.side_effect=response
        app=AppTest.from_file(str(APP),default_timeout=20)
        app.session_state["auth"]=session
        app.run()
        app.chat_input[0].set_value("Dinh Độc Lập ở đâu?").run()
        self.assertFalse(app.exception)
        self.assertIn("[G01]",app.session_state["messages"][-1]["content"])
        app.chat_input[0].set_value("Địa chỉ Dinh Độc Lập là gì?").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["messages"]),4)
        full_guide_calls=[call for call in post.call_args_list if call.args[0].endswith("/api/embed") and len(call.kwargs["json"]["input"])==16]
        self.assertEqual(len(full_guide_calls),1)
        cache_resource.clear()


if __name__ == "__main__": unittest.main()
