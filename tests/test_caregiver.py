"""No network, no phone: a fake Photon link plays the caregiver; notify() talks to a local test server.
    python3 -m unittest discover -s tests
"""

import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eyereach import notify as notify_module  # noqa: E402
from services.photon.caregiver import Diary, Service, answer, summary  # noqa: E402


class FakeLink:
    """Photon stand-in: texts waiting from the caregiver, and everything we sent."""
    def __init__(self, chat=True):
        self.chat = {"spaceId": "s1", "senderId": "agent"} if chat else None
        self.inbox, self.sent = [], []

    def incoming(self):
        texts, self.inbox = self.inbox, []
        return texts

    def send(self, text, reply_to=None):
        if not self.chat:
            return False
        self.sent.append((text, reply_to))
        return True


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.dir.name)
        self.diary = Diary(self.folder / "events.jsonl")
        self.stop = self.folder / "STOP"

    def tearDown(self):
        self.dir.cleanup()


class AnswerTest(Base):
    def test_stop_and_resume(self):
        self.assertIn("won't move", answer("STOP", self.diary, self.stop))
        self.assertTrue(self.stop.exists())
        self.assertIn("STOPPED", answer("status", self.diary, self.stop))
        self.assertIn("move again", answer(" Resume! ", self.diary, self.stop))
        self.assertFalse(self.stop.exists())

    def test_stop_never_goes_to_the_ai(self):
        def ai(*args):
            raise AssertionError("STOP must not reach the AI")
        answer("stop", self.diary, self.stop, ask_ai=ai)
        self.assertTrue(self.stop.exists())

    def test_status_lists_todays_robot_events(self):
        self.diary.add("picked", "Picked up the pill bottle.")
        self.diary.add("caregiver", "hi", who="caregiver")
        text = answer("What happened today?", self.diary, self.stop)
        self.assertIn("Picked up the pill bottle.", text)
        self.assertNotIn("hi", text.split(":", 1)[1])

    def test_free_questions_go_to_the_ai_with_the_log(self):
        self.diary.add("picked", "Picked up the pill bottle.")
        seen = {}

        def ai(question, events, stopped):
            seen.update(question=question, events=events, stopped=stopped)
            return "Yes, at 10:02."
        self.assertEqual(answer("Did he take his pills?", self.diary, self.stop, ask_ai=ai), "Yes, at 10:02.")
        self.assertEqual(seen["events"][0]["text"], "Picked up the pill bottle.")
        self.assertFalse(seen["stopped"])

    def test_ai_failure_falls_back_to_help(self):
        def ai(*args):
            raise TimeoutError("no network")
        self.assertIn("STATUS", answer("how is he?", self.diary, self.stop, ask_ai=ai))


class DiaryTest(Base):
    def test_only_today_counts(self):
        yesterday = (datetime.now() - timedelta(days=1)).isoformat(timespec="seconds")
        self.diary.path.write_text(json.dumps({"time": yesterday, "who": "robot", "event": "x", "text": "old"}) + "\n")
        self.diary.add("picked", "new")
        self.assertEqual([e["text"] for e in self.diary.today()], ["new"])

    def test_summary_keeps_the_last_eight(self):
        events = [{"time": f"2026-11-14T10:{i:02d}:00", "who": "robot", "event": "e", "text": f"event {i}"}
                  for i in range(12)]
        text = summary(events, stopped=False)
        self.assertIn("(4 earlier)", text)
        self.assertIn("event 11", text)
        self.assertNotIn("event 3\n", text)


class ServiceTest(Base):
    def service(self, link):
        return Service(link, self.diary, self.stop)

    def test_robot_events_are_logged_and_texted(self):
        link = FakeLink()
        self.service(link).robot_says("picked", "Picked up the cup.")
        self.assertEqual(link.sent, [("Picked up the cup.", None)])
        self.assertEqual(self.diary.today()[0]["text"], "Picked up the cup.")

    def test_before_the_caregiver_texts_first_nothing_is_sent_but_it_is_logged(self):
        link = FakeLink(chat=False)
        with self.assertLogs("caregiver", "WARNING"):
            self.service(link).robot_says("picked", "Picked up the cup.")
        self.assertEqual(link.sent, [])
        self.assertEqual(len(self.diary.today()), 1)

    def test_caregiver_stop_by_text(self):
        link = FakeLink()
        link.inbox = [("m1", "STOP")]
        self.service(link).check_texts()
        self.assertTrue(self.stop.exists())
        self.assertEqual(link.sent[0][1], "m1")                         # a reply to their message
        self.assertIn("RESUME", link.sent[0][0])


class NotifyTest(Base):
    def test_notify_reaches_the_service(self):
        link = FakeLink()
        server = self.service_server(link)
        try:
            notify_module.notify("picked", "Picked up the pill bottle.",
                                 url=f"http://127.0.0.1:{server.server_address[1]}/notify").join(5)
            for _ in range(50):                                          # the service sends just after answering
                if link.sent:
                    break
                time.sleep(0.02)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(link.sent, [("Picked up the pill bottle.", None)])

    def service_server(self, link):
        return Service(link, self.diary, self.stop).serve(port=0)

    def test_notify_never_blocks_or_raises_when_the_service_is_down(self):
        began = time.monotonic()
        thread = notify_module.notify("picked", "x", url="http://127.0.0.1:9/notify")   # nothing listens on 9
        self.assertLess(time.monotonic() - began, 0.1)                   # the caller is not kept waiting
        thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_stopped_reads_the_stop_file(self):
        original = notify_module.STOP_FILE
        notify_module.STOP_FILE = self.stop
        try:
            self.assertFalse(notify_module.stopped())
            self.stop.write_text("now")
            self.assertTrue(notify_module.stopped())
        finally:
            notify_module.STOP_FILE = original


if __name__ == "__main__":
    unittest.main()
