import time
import unittest
from unittest.mock import patch

from firebase_client import AuthError, AuthSession, FirebaseClient, clear_session


class AuthTests(unittest.TestCase):
    @patch("firebase_client.requests.post")
    def test_signin_calls_firebase_and_retains_no_password(self, post):
        post.return_value.ok = True
        post.return_value.json.return_value = {"email": "test@example.com", "localId": "u1", "idToken": "id", "refreshToken": "refresh", "expiresIn": "3600"}
        session = FirebaseClient("test-key").authenticate(" test@example.com ", "test-password")
        self.assertEqual(session.uid, "u1")
        self.assertNotIn("password", vars(session))
        self.assertTrue(post.call_args.args[0].endswith("accounts:signInWithPassword"))
        self.assertTrue(post.call_args.kwargs["json"]["returnSecureToken"])

    @patch("firebase_client.requests.post")
    def test_wrong_credentials_have_a_clear_error(self, post):
        post.return_value.ok = False
        post.return_value.json.return_value = {"error": {"message": "INVALID_LOGIN_CREDENTIALS"}}
        with self.assertRaisesRegex(AuthError, "Email hoặc mật khẩu"):
            FirebaseClient("test-key").authenticate("test@example.com", "wrong")

    @patch("firebase_client.requests.post")
    def test_expired_session_refreshes_then_validates_the_user(self, post):
        payloads = [{"user_id":"u1","id_token":"new-id","refresh_token":"new-refresh","expires_in":"3600"},
                    {"users":[{"localId":"u1"}]}]
        post.return_value.ok = True
        post.return_value.json.side_effect = payloads
        result = FirebaseClient("test-key").validate(AuthSession("a@b.com", "u1", "old-id", "refresh", 0))
        self.assertEqual(result.id_token, "new-id")
        self.assertEqual(post.call_count, 2)

    def test_signout_clears_chat_and_tokens(self):
        state = {"auth": object(), "messages": ["private conversation"], "login_password": "not retained"}
        clear_session(state)
        self.assertEqual(state, {})

    @patch("firebase_client.requests.post")
    def test_signup_short_password_does_not_call_firebase(self, post):
        with self.assertRaises(AuthError): FirebaseClient("test-key").authenticate("a@b.com", "123", True)
        post.assert_not_called()


if __name__ == "__main__": unittest.main()
