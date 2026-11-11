import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__": unittest.main()
