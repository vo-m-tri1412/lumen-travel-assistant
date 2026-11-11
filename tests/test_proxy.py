import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock, patch

from server.proxy import make_handler


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.token="test-token-for-local-unit-tests-only"
        self.server=ThreadingHTTPServer(("127.0.0.1",0),make_handler(self.token))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self,body,token=None,path="/api/embed"):
        connection=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=5)
        headers={"Content-Type":"application/json"}
        if token:headers["Authorization"]="Bearer "+token
        connection.request("POST",path,json.dumps(body),headers)
        response=connection.getresponse()
        status,data=response.status,json.loads(response.read())
        connection.close()
        return status,data

    @patch("server.proxy.urlopen")
    def test_unauthenticated_requests_never_reach_ollama(self,upstream):
        status,_=self.request({"model":"embeddinggemma","input":["hello"]})
        self.assertEqual(status,401)
        upstream.assert_not_called()

    @patch("server.proxy.urlopen")
    def test_model_management_and_wrong_models_are_not_exposed(self,upstream):
        self.assertEqual(self.request({"model":"other-model"},self.token)[0],400)
        self.assertEqual(self.request({"name":"anything"},self.token,"/api/pull")[0],404)
        upstream.assert_not_called()

    @patch("server.proxy.urlopen")
    def test_authenticated_embedding_request_is_forwarded(self,upstream):
        upstream.return_value.__enter__.return_value.status=200
        upstream.return_value.__enter__.return_value.read.return_value=b'{"embeddings":[[1,0]]}'
        status,data=self.request({"model":"embeddinggemma","input":["hello"]},self.token)
        self.assertEqual(status,200)
        self.assertEqual(data["embeddings"],[[1,0]])
        self.assertNotIn(self.token,str(upstream.call_args))


if __name__=="__main__":unittest.main()
