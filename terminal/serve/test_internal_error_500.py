"""An exception the handler does not expect (a FileNotFoundError from a cached image file deleted under the server) is
answered 500 with its name, instead of a dropped connection the client can only report as ECONNRESET; a client that
left (ConnectionError) is still re-raised, and an answer already begun is not replaced.

    python -m serve.test_internal_error_500
"""
import unittest
from types import SimpleNamespace

import serve.server as server


def handler(raise_this, answer_started=False):
    svc = SimpleNamespace(begin_request=lambda path, req: None, drop_embeddings=lambda: None, reserve_inference=lambda: True, release_inference=lambda: None)
    H = server.make_handler(svc)
    h = H.__new__(H)
    h.path = "/v1/chat/completions"
    h.record, h.watch_done, h.answer_started = None, None, answer_started
    h._authorized = lambda: True
    h._foreign_page = lambda: False
    h._body = lambda: b'{"messages": []}'
    h.sent = []
    h._json = lambda code, body, headers=None: h.sent.append((code, body))

    def openai(req):
        raise raise_this
    h._openai = openai
    return h


class InternalError(unittest.TestCase):
    def test_unexpected_oserror_is_a_500(self):
        h = handler(FileNotFoundError(2, "No such file", "x.sve"))
        h._post()
        self.assertEqual(len(h.sent), 1)
        code, body = h.sent[0]
        self.assertEqual(code, 500)
        self.assertIn("FileNotFoundError", body["error"]["message"])

    def test_other_exception_is_a_500(self):
        h = handler(RuntimeError("boom"))
        h._post()
        self.assertEqual(h.sent[0][0], 500)
        self.assertIn("boom", h.sent[0][1]["error"]["message"])

    def test_client_gone_is_still_raised(self):
        h = handler(ConnectionResetError())
        with self.assertRaises(ConnectionResetError):
            h._post()
        self.assertEqual(h.sent, [])

    def test_begun_answer_is_not_replaced(self):
        h = handler(RuntimeError("late"), answer_started=True)
        with self.assertRaises(RuntimeError):
            h._post()
        self.assertEqual(h.sent, [])

    def test_bad_body_is_left_to_do_post(self):
        h = handler(server.BadBody(413, "too big"))
        with self.assertRaises(server.BadBody):
            h._post()
        self.assertEqual(h.sent, [])

    def test_value_error_keeps_its_400(self):
        h = handler(ValueError("bad"))
        h._post()
        self.assertEqual(h.sent[0][0], 400)


if __name__ == "__main__":
    unittest.main()
