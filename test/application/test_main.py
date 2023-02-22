import unittest
from starlette.testclient import TestClient
from application.main import app


class TestMain(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_main(self):
        response = self.client.get("/image-captioning/")
        self.assertEqual(response.status_code, 200)

    def test_caption_image(self):
        response = self.client.post("/image-captioning/caption_image/")
        self.assertEqual(response.status_code, 422)

    def test_upload_image(self):
        response = self.client.post("/image-captioning/upload_image/")
        self.assertEqual(response.status_code, 422)
