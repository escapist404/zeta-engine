import unittest

from zeta_engine.utils import url_normalize


class NormalizeUrlTest(unittest.TestCase):
    def test_normalizes_common_http_urls(self) -> None:
        cases = {
            " HTTPS://Example.COM:443#intro ": "https://example.com/",
            "http://Example.COM:80/a?X=Y#frag": "http://example.com/a?X=Y",
            "https://Example.COM:8443/a": "https://example.com:8443/a",
            "https://[2001:DB8::1]:443": "https://[2001:db8::1]/",
        }

        for raw_url, expected in cases.items():
            with self.subTest(raw_url=raw_url):
                self.assertEqual(url_normalize(raw_url), expected)

    def test_preserves_case_sensitive_userinfo(self) -> None:
        self.assertEqual(
            url_normalize(
                "HTTPS://Alice:S3CrEt@Example.COM:443/private"
            ),
            "https://Alice:S3CrEt@example.com/private",
        )

    def test_only_removes_ports_for_exact_http_schemes(self) -> None:
        cases = {
            "https+unix://Example.COM:443/socket": (
                "https+unix://example.com:443/socket"
            ),
            "httpx://Example.COM:80/resource": (
                "httpx://example.com:80/resource"
            ),
        }

        for raw_url, expected in cases.items():
            with self.subTest(raw_url=raw_url):
                self.assertEqual(url_normalize(raw_url), expected)

    def test_removes_zero_padded_default_ports(self) -> None:
        cases = {
            "https://Example.COM:00443/a": "https://example.com/a",
            "http://Example.COM:00080/a": "http://example.com/a",
        }

        for raw_url, expected in cases.items():
            with self.subTest(raw_url=raw_url):
                self.assertEqual(url_normalize(raw_url), expected)

    def test_normalization_is_idempotent(self) -> None:
        urls = (
            " HTTPS://Example.COM:443#intro ",
            "http://Example.COM:80/a?X=Y#frag",
            "https://Example.COM:8443/a",
            "https://[2001:DB8::1]:443",
        )

        for raw_url in urls:
            with self.subTest(raw_url=raw_url):
                once = url_normalize(raw_url)
                self.assertEqual(url_normalize(once), once)


if __name__ == "__main__":
    unittest.main()
