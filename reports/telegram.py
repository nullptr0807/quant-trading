"""Send reports to Telegram via Bot API using only urllib."""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone


CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
API_BASE = "https://api.telegram.org/bot{token}"


class TelegramReporter:
    """Send text messages and photos to Telegram."""

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or os.environ["TELEGRAM_BOT_TOKEN"]
        self.chat_id = chat_id or CHAT_ID
        if not self.chat_id:
            raise RuntimeError("TELEGRAM_CHAT_ID env var not set")
        self.base_url = API_BASE.format(token=self.token)

    # -- public helpers ------------------------------------------------

    def send_report(self, text: str, image_path: str | None = None, force: bool = False):
        """Send a full report: text message + optional chart photo."""
        if not force:
            now_utc = datetime.now(timezone.utc)
            # Only send during US extended hours Mon-Fri 08:00-01:00 UTC
            if now_utc.weekday() >= 5:
                return
            hour = now_utc.hour
            if 1 <= hour < 8:
                return
        self.send_message(text)
        if image_path and os.path.isfile(image_path):
            self.send_photo(image_path, caption="\U0001F4C8 Equity curves")

    # -- core methods --------------------------------------------------

    def send_message(self, text: str):
        """Send a text message via Telegram Bot API. Auto-splits if >4096 chars."""
        chunks = []
        while len(text) > 4096:
            # Find last newline before 4096
            split_at = text.rfind('\n', 0, 4096)
            if split_at == -1:
                split_at = 4096
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip('\n')
        chunks.append(text)

        results = []
        for chunk in chunks:
            url = f"{self.base_url}/sendMessage"
            payload = json.dumps({
                "chat_id": self.chat_id,
                "text": chunk,
            }).encode()
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    results.append(json.loads(resp.read()))
            except Exception as exc:
                print(f"[TelegramReporter] send_message failed: {exc}")
                results.append(None)
        return results[-1] if results else None

    def send_photo(self, image_path: str, caption: str = ""):
        """Send a photo via multipart/form-data upload."""
        url = f"{self.base_url}/sendPhoto"
        boundary = "----PythonFormBoundary"
        body = bytearray()

        # chat_id field
        body += f"--{boundary}\r\n".encode()
        body += b'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
        body += f"{self.chat_id}\r\n".encode()

        # caption field
        if caption:
            body += f"--{boundary}\r\n".encode()
            body += b'Content-Disposition: form-data; name="caption"\r\n\r\n'
            body += f"{caption}\r\n".encode()

        # photo file
        filename = os.path.basename(image_path)
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="photo"; filename="{filename}"\r\n'.encode()
        body += b"Content-Type: image/png\r\n\r\n"
        with open(image_path, "rb") as f:
            body += f.read()
        body += b"\r\n"
        body += f"--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            url, data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            print(f"[TelegramReporter] send_photo failed: {exc}")
            return None
