import unittest
from unittest.mock import MagicMock, patch
import mailer
import config

class TestMailer(unittest.TestCase):

    @patch('mailer.smtplib.SMTP')
    def test_send_report(self, mock_smtp):
        # Setup
        m = mailer.Mailer()
        # Mock config values if they are None in test env
        m.user = "test@example.com"
        m.password = "pass"
        m.receiver = "recv@example.com"

        md_content = "# Report\n\n* Item 1\n* Item 2"
        date_str = "2023-10-27"

        # Action
        m.send_report(md_content, date_str)

        # Verify
        mock_smtp.assert_called_with("smtp.gmail.com", 587)
        instance = mock_smtp.return_value
        # When using 'with', the object used is what __enter__ returns
        server_mock = instance.__enter__.return_value
        server_mock.starttls.assert_called()
        server_mock.login.assert_called_with("test@example.com", "pass")
        server_mock.sendmail.assert_called()

        # Verify HTML conversion roughly
        call_args = server_mock.sendmail.call_args
        msg_str = call_args[0][2] # The message string
        self.assertIn("Subject: Edge AI Scout Report - 2023-10-27", msg_str)
        self.assertIn("<html>", msg_str)
        self.assertIn("Item 1", msg_str)
        self.assertIn("font-family: Arial", msg_str)

    def test_html_conversion(self):
        m = mailer.Mailer()
        md = "# Hello\n\nThis is a **test**."
        html = m.convert_markdown_to_html(md)

        self.assertIn("<h1>Hello</h1>", html)
        self.assertIn("<strong>test</strong>", html)
        self.assertIn("<style>", html)
        self.assertIn("body {", html)

if __name__ == '__main__':
    unittest.main()
