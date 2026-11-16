"""#5083: the DFlash2 drafter download takes the Hugging Face host from HF_ENDPOINT (a mirror such as hf-mirror.com), as setup and
huggingface_hub do; read at each fetch, same pinned revision and SHA-256 checks. Without it: huggingface.co."""

import os
import unittest
from unittest import mock

from strata_control import drafting, drafting_fetch


class DrafterHost(unittest.TestCase):
    def test_hf_endpoint_names_the_host(self):
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.example/"}):
            self.assertEqual(drafting.base_url(), f"https://hf-mirror.example/{drafting.REPO}/resolve/{drafting.REVISION}/")

    def test_without_it_huggingface_co(self):
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": ""}):
            self.assertEqual(drafting.base_url(), drafting.BASE_URL)
            self.assertTrue(drafting.BASE_URL.startswith("https://huggingface.co/"))

    def test_the_fetch_takes_the_address_at_fetch_time(self):
        import inspect
        source = inspect.getsource(drafting_fetch)
        self.assertIn("drafting.base_url()", source)
        self.assertNotIn("drafting.BASE_URL", source)


if __name__ == "__main__":
    unittest.main()
