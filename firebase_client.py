"""Firebase email/password authentication through its documented REST API."""

import time
from dataclasses import dataclass, field

import requests


class AuthError(Exception):
    pass


@dataclass
class AuthSession:
    email: str
    uid: str
    id_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: float = 0


class FirebaseClient:
    def __init__(self, api_key):
        if not api_key:
            raise AuthError("Ứng dụng chưa được cấu hình đăng nhập. Xem README để thiết lập.")
        self.api_key = api_key

    def _post(self, endpoint, payload, refresh=False):
        host = "https://securetoken.googleapis.com/v1/token" if refresh else (
            "https://identitytoolkit.googleapis.com/v1/accounts:" + endpoint
        )
        try:
            response = requests.post(host, params={"key": self.api_key},
                                     timeout=(10, 25), **({"data": payload} if refresh else {"json": payload}))
            result = response.json()
        except (requests.RequestException, ValueError) as error:
            raise AuthError("Chưa kết nối được dịch vụ đăng nhập. Vui lòng thử lại.") from error
        if not isinstance(result, dict):
            raise AuthError("Dịch vụ đăng nhập trả về dữ liệu không hợp lệ. Vui lòng thử lại.")
        if not response.ok:
            code = str(result.get("error", {}).get("message", "UNKNOWN")).split(" : ")[0]
            messages = {
                "EMAIL_EXISTS": "Email này đã có tài khoản. Bạn hãy đăng nhập.",
                "INVALID_EMAIL": "Địa chỉ email chưa hợp lệ.",
                "WEAK_PASSWORD": "Mật khẩu cần ít nhất 6 ký tự.",
                "OPERATION_NOT_ALLOWED": "Dự án Firebase chưa bật đăng nhập Email/Password.",
                "TOO_MANY_ATTEMPTS_TRY_LATER": "Bạn thử quá nhiều lần. Vui lòng chờ rồi thử lại.",
                "USER_DISABLED": "Tài khoản này đang bị vô hiệu hóa.",
                "INVALID_LOGIN_CREDENTIALS": "Email hoặc mật khẩu chưa đúng.",
                "EMAIL_NOT_FOUND": "Email hoặc mật khẩu chưa đúng.",
                "INVALID_PASSWORD": "Email hoặc mật khẩu chưa đúng.",
                "TOKEN_EXPIRED": "Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.",
                "INVALID_ID_TOKEN": "Phiên đăng nhập không hợp lệ. Vui lòng đăng nhập lại.",
                "INVALID_REFRESH_TOKEN": "Phiên đăng nhập không hợp lệ. Vui lòng đăng nhập lại.",
            }
            raise AuthError(messages.get(code, "Không thể đăng nhập. Kiểm tra cấu hình Firebase hoặc thử lại."))
        return result

    def authenticate(self, email, password, signup=False):
        email = email.strip()
        if not email or not password:
            raise AuthError("Bạn cần nhập email và mật khẩu.")
        if signup and len(password) < 6:
            raise AuthError("Mật khẩu cần ít nhất 6 ký tự.")
        data = self._post("signUp" if signup else "signInWithPassword",
                          {"email": email, "password": password, "returnSecureToken": True})
        return AuthSession(data["email"], data["localId"], data["idToken"],
                           data["refreshToken"], time.time() + int(data["expiresIn"]))

    def validate(self, session):
        if session.expires_at <= time.time() + 60:
            data = self._post("token", {"grant_type": "refresh_token", "refresh_token": session.refresh_token}, True)
            session = AuthSession(session.email, data["user_id"], data["id_token"],
                                  data["refresh_token"], time.time() + int(data["expires_in"]))
        account = self._post("lookup", {"idToken": session.id_token})
        users = account.get("users", [])
        if not users or users[0].get("localId") != session.uid or users[0].get("disabled", False):
            raise AuthError("Phiên đăng nhập không còn hợp lệ. Vui lòng đăng nhập lại.")
        return session


def clear_session(state):
    """Sign-out removes history, credentials, and input widgets for this session."""
    for name in list(state):
        del state[name]
