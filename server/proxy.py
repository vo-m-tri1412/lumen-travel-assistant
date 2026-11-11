"""A small authenticated proxy for exposing only the required Ollama methods."""

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def make_handler(token, upstream="http://127.0.0.1:11434"):
    if len(token) < 24:
        raise ValueError("OLLAMA_API_TOKEN needs at least 24 characters.")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Do not log credentials, questions, or guide content.

        def reply(self, status, content):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def forward(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer "+token):
                self.reply(401, b'{"error":"Authentication required"}')
                return
            required = {("POST","/api/embed"):"embeddinggemma", ("POST","/api/chat"):"qwen3:4b"}
            expected_model = required.get((self.command, self.path))
            if (self.command,self.path)!=("GET","/api/tags") and expected_model is None:
                self.reply(404, b'{"error":"Route not available"}')
                return
            body = None
            if self.command == "POST":
                try:
                    length=int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 2_000_000:
                        raise ValueError()
                    payload=json.loads(self.rfile.read(length))
                    if not isinstance(payload,dict) or payload.get("model")!=expected_model or payload.get("stream") is True:
                        raise ValueError()
                    body=json.dumps(payload).encode("utf-8")
                except (ValueError, TypeError):
                    self.reply(400,b'{"error":"Invalid model or request body"}')
                    return
            try:
                request=Request(upstream+self.path,data=body,method=self.command,headers={"Content-Type":"application/json"})
                with urlopen(request,timeout=180) as response:
                    self.reply(response.status,response.read())
            except HTTPError as error:
                self.reply(error.code,error.read())
            except (URLError,TimeoutError):
                self.reply(503,b'{"error":"Ollama is unavailable"}')

        do_GET=forward
        do_POST=forward

    return Handler


if __name__ == "__main__":
    secret=os.environ.get("OLLAMA_API_TOKEN", "")
    print("Authenticated Ollama proxy: 127.0.0.1:11435",flush=True)
    ThreadingHTTPServer(("127.0.0.1",11435),make_handler(secret)).serve_forever()
