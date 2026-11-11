import unittest
from server.tunnel import extract_tunnel_url


class TunnelTests(unittest.TestCase):
    def test_dashboard_banner_is_not_an_api_endpoint(self):
        self.assertIsNone(extract_tunnel_url("Manage at https://dashboard.pinggy.io"))

    def test_actual_tunnel_is_selected_among_banner_links(self):
        banner="https://dashboard.pinggy.io  https://sample.a.free.pinggy.link"
        self.assertEqual(extract_tunnel_url(banner),"https://sample.a.free.pinggy.link")

    def test_terminal_color_codes_do_not_split_the_url(self):
        self.assertEqual(extract_tunnel_url("https://sample.a.free.\x1b[34mpinggy.link\x1b[0m"),"https://sample.a.free.pinggy.link")


if __name__=="__main__":unittest.main()
