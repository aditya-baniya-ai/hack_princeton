"""The caregiver's iMessage agent, on Photon 2.0 (beta). Two-way:

    the robot -> caregiver   eyereach.notify.notify() posts to http://127.0.0.1:8787/notify; the text is logged
                             and sent to the caregiver on iMessage.
    caregiver -> the agent   STOP      pauses the arm (writes ~/.eyereach/STOP, which eyereach.notify.stopped() reads)
                             RESUME    lets it move again
                             STATUS    or "what happened today?": today's events, from the event log
                             anything else ("did he take his pills?") is answered by Nova from the event log

STOP and RESUME are plain code, never the AI, so they always work the same way. Every event and message is kept in
~/.eyereach/events.jsonl, so the agent remembers the whole day.

    python3 -m services.photon.caregiver

Photon's shared iMessage line can't start a conversation: the caregiver texts the agent's number once (e.g. "hi")
before the robot can message them. The service remembers that conversation in ~/.eyereach/photon_chat.json.

Environment variables:
    PHOTON_API_KEY, PHOTON_PROJECT_ID     the Photon project (key with project:read and project:write)
    PHOTON_AGENT_HANDLE                   the agent's iMessage number Photon assigned, e.g. +16095550123
    PHOTON_CAREGIVER                      the caregiver's iMessage number (or Apple-account email)
    EYEREACH_NOTIFY_PORT                  default 8787
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, NOVA_MODEL_ID   for answering free questions (optional)
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from eyereach.notify import FOLDER, STOP_FILE

log = logging.getLogger("caregiver")

EVENTS = FOLDER / "events.jsonl"
CHAT = FOLDER / "photon_chat.json"
PORT = int(os.environ.get("EYEREACH_NOTIFY_PORT", "8787"))
POLL = 2.0                                     # seconds between checks for the caregiver's texts
HELP = "Text STATUS for today's events, STOP to pause the arm, or RESUME to let it move again."


# ---- the event log: one JSON line per event, kept across the day -------------------------------------------

class Diary:
    def __init__(self, path=EVENTS):
        self.path = path

    def add(self, event, text, who="robot"):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = {"time": datetime.now().isoformat(timespec="seconds"), "who": who, "event": event, "text": text}
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def today(self):
        if not self.path.exists():
            return []
        day = datetime.now().date().isoformat()
        rows = (json.loads(line) for line in self.path.read_text().splitlines() if line.strip())
        return [r for r in rows if r["time"].startswith(day)]


def summary(events, stopped):
    """Today's robot events (and STOP/RESUME), newest last, as a short text."""
    shown = [e for e in events if e["who"] == "robot" or e["event"] in ("stop", "resume")]
    if not shown:
        text = "Nothing has happened yet today."
    else:
        lines = [f"{e['time'][11:16]} {e['text']}" for e in shown[-8:]]
        more = f" ({len(shown) - 8} earlier)" if len(shown) > 8 else ""
        text = f"Today{more}:\n" + "\n".join(lines)
    return text + ("\nThe arm is STOPPED. Text RESUME to let it move." if stopped else "")


# ---- what to answer -------------------------------------------------------------------------------------

def answer(text, diary, stop_file=STOP_FILE, ask_ai=None):
    """The reply to one caregiver text. STOP / RESUME / STATUS are fixed; anything else goes to the AI."""
    said = text.strip().lower().strip(" .!?")
    if said == "stop":
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.write_text(datetime.now().isoformat(timespec="seconds"))
        diary.add("stop", "The caregiver sent STOP.", who="caregiver")
        return "Stopped. The arm won't move until you text RESUME."
    if said == "resume":
        stop_file.unlink(missing_ok=True)
        diary.add("resume", "The caregiver sent RESUME.", who="caregiver")
        return "Resumed. The arm can move again."
    if said in ("status", "what happened today", "what happened", "update"):
        return summary(diary.today(), stop_file.exists())
    if said in ("help", "hi", "hello", ""):
        return f"Hi! I'm the EyeReach assistant. {HELP}"
    if ask_ai:
        try:
            return ask_ai(text.strip(), diary.today(), stop_file.exists())
        except Exception as e:
            log.warning("AI answer failed: %s", e)
    return f"Sorry, I didn't understand. {HELP}"


def ask_nova(question, events, stopped):
    """Nova answers a free question from today's event log (text only, one short call)."""
    import boto3
    from botocore.config import Config
    lines = "\n".join(f"{e['time'][11:16]} [{e['who']}] {e['text']}" for e in events[-60:]) or "(no events yet)"
    prompt = (f"You text a caregiver about a person who uses a robot arm controlled with their eyes. "
              f"Answer the caregiver's message in one or two short sentences, using ONLY this log of today. "
              f"If the log doesn't say, say you don't know. The arm is {'STOPPED' if stopped else 'allowed to move'}.\n"
              f"Log:\n{lines}\n\nCaregiver: {question}")
    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"),
                          config=Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 1}))
    response = client.converse(modelId=os.environ.get("NOVA_MODEL_ID", "us.amazon.nova-2-lite-v1:0"),
                               messages=[{"role": "user", "content": [{"text": prompt}]}],
                               inferenceConfig={"temperature": 0, "maxTokens": 200})
    return "".join(p.get("text", "") for p in response["output"]["message"]["content"]).strip()


# ---- Photon: read the caregiver's texts, send ours --------------------------------------------------------

class PhotonLink:
    def __init__(self, key, project, agent, caregiver, chat_file=CHAT):
        from photon_api import Photon
        self.photon = Photon(headers={"Authorization": f"Bearer {key}"})
        self.project, self.agent, self.caregiver, self.chat_file = project, agent, caregiver, chat_file
        self.chat = json.loads(chat_file.read_text()) if chat_file.exists() else None
        self.since = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(timespec="milliseconds")
        self.seen = set()

    def incoming(self):
        """The caregiver's new texts to the agent, oldest first: [(message id, text)]."""
        from photon_api.generated.models import TextContent, UserSnapshot
        from photon_api.rpc_generated import ListMessagesInput
        query = {"createdSince": self.since.replace("+00:00", "Z"), "direction": "received", "order": "asc",
                 "pageSize": 100, "platform": "imessage"}
        found = []
        while True:
            page = self.photon.projects.list_messages(
                ListMessagesInput.model_validate({"path": {"id": self.project}, "query": query}))
            for item in page.items:
                m = item.root.root
                if m.id in self.seen or m.status != "received" or not isinstance(m.receivedBy, UserSnapshot):
                    continue
                sender, agent = m.sentBy.root, m.receivedBy.root
                if sender.kind != "human" or sender.handle != self.caregiver or agent.handle != self.agent:
                    continue
                self.seen.add(m.id)
                self.remember(m.space.root.id, agent.id)
                if isinstance(m.content, TextContent):
                    found.append((m.id, m.content.text))
            token = page.nextPageToken
            if not isinstance(token, str):
                return found
            query["pageToken"] = token

    def remember(self, space_id, agent_id):
        chat = {"spaceId": space_id, "senderId": agent_id}
        if chat != self.chat:
            self.chat = chat
            self.chat_file.parent.mkdir(parents=True, exist_ok=True)
            self.chat_file.write_text(json.dumps(chat))

    def send(self, text, reply_to=None):
        """Text the caregiver. False if they haven't texted the agent yet (Photon can't start the chat)."""
        if not self.chat:
            return False
        from photon_api.generated.models import SendMessageRequest, SendTextContent
        from photon_api.rpc_generated import SendMessageHeader, SendMessageInput, SendMessagePath
        body = {"spaceId": self.chat["spaceId"], "senderId": self.chat["senderId"],
                "content": SendTextContent(type="text", format="plain", text=text)}
        if reply_to:
            body["parentMessageId"] = reply_to
        self.photon.projects.send_message(SendMessageInput(
            path=SendMessagePath(id=self.project),
            headers=SendMessageHeader.model_validate({"idempotency-key": f"eyereach:{reply_to or uuid.uuid4()}"}),
            body=SendMessageRequest(**body)))
        return True


# ---- the service ----------------------------------------------------------------------------------------

class Service:
    def __init__(self, link, diary=None, stop_file=STOP_FILE, ask_ai=None):
        self.link, self.diary, self.stop_file, self.ask_ai = link, diary or Diary(), stop_file, ask_ai
        self.lock = threading.Lock()

    def robot_says(self, event, text):
        """From notify(): log it and text the caregiver."""
        with self.lock:
            self.diary.add(event, text)
            try:
                if not self.link.send(text):
                    log.warning("Not sent: the caregiver must text the agent's number once first.")
            except Exception as e:
                log.warning("Could not send to the caregiver: %s", e)

    def check_texts(self):
        """Answer every new caregiver text."""
        for message_id, text in self.link.incoming():
            with self.lock:
                self.diary.add("caregiver", text, who="caregiver")
                reply = answer(text, self.diary, self.stop_file, self.ask_ai)
                self.diary.add("answer", reply, who="agent")
                self.link.send(reply, reply_to=message_id)
            log.info("caregiver: %r -> %r", text, reply)

    def serve(self, port=PORT):
        """The /notify endpoint, in a background thread. Returns the server."""
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/notify":
                    self.send_error(404)
                    return
                try:
                    data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    event, text = str(data.get("event", "event")), str(data["text"])
                except (ValueError, KeyError):
                    self.send_error(400, "expected JSON with 'text'")
                    return
                self.send_response(202)                 # answer at once; sending to Photon happens after
                self.end_headers()
                threading.Thread(target=service.robot_says, args=(event, text), daemon=True).start()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def run(self, port=PORT):
        server = self.serve(port)
        log.info("Listening on http://127.0.0.1:%d/notify; checking the caregiver's texts every %.0f s.",
                 server.server_address[1], POLL)
        if not self.link.chat:
            log.info("Waiting for the caregiver to text the agent's number once (Photon can't start the chat).")
        while True:
            try:
                self.check_texts()
            except Exception as e:                      # network blips: keep going
                log.warning("Checking texts failed: %s", e)
            time.sleep(POLL)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)       # not a line for every 2-second check
    names = ("PHOTON_API_KEY", "PHOTON_PROJECT_ID", "PHOTON_AGENT_HANDLE", "PHOTON_CAREGIVER")
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} (see the README).")
    link = PhotonLink(*(os.environ[n] for n in names))
    Service(link, ask_ai=ask_nova).run()


if __name__ == "__main__":
    main()
