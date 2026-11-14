"""The task agent inside the eye app: "bring me the banana" said to Nova 2 Sonic becomes a task for
agent/task_agent.py --serve, which runs in the agent's Python 3.12 (deepagents) as its own program.

The app keeps the arm: the agent only asks (confirm this object? pick at this spot of the table camera's
picture?) and the app answers, so the camera, the buttons and STOP keep working while it thinks and moves.
"""

import json
import queue
import subprocess
import sys
import threading

from eyereach.voice_commands import AGENT_PYTHON, REPO


class AgentLink:
    def __init__(self, table_camera, python=AGENT_PYTHON):
        self.table_camera, self.python = table_camera, python
        self.messages = queue.Queue()
        self.process = None
        self.status = "Starting the agent"

    def start(self):
        try:
            self.process = subprocess.Popen(
                [str(self.python), "-u", "-m", "agent.task_agent", "--serve", "--camera", str(self.table_camera)],
                cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True)
        except OSError as error:
            self.status = f"The agent did not start: {error}"
            return self
        threading.Thread(target=self.read, args=(self.process.stdout,), name="eyereach-agent", daemon=True).start()
        return self

    def read(self, lines):
        for line in lines:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if "status" in message:
                self.status = message["status"]
            self.messages.put(message)
        self.status = "The agent stopped"
        self.messages.put({"reply": "The task agent stopped."})     # ends a task that was running

    def send(self, **message):
        """False if the agent isn't running."""
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
            return True
        except (AttributeError, OSError, ValueError):
            return False

    def poll(self):
        items = []
        while True:
            try:
                items.append(self.messages.get_nowait())
            except queue.Empty:
                return items

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()


def run_agent_messages(link, ctl, to_job, now):
    """One frame of the conversation between the app's control (ctl) and the agent.
    to_job(u, v): a FetchJob for this spot of the table camera's picture."""
    if ctl.wants_task:
        task, ctl.wants_task = ctl.wants_task, None
        if link.send(task=task):
            ctl.task_running = True
        else:
            ctl.task_done("The task agent isn't running.", now)
    for answer in ctl.answers:
        link.send(answer=answer)
    ctl.answers.clear()
    for result in ctl.fetched:
        link.send(result=result)
    ctl.fetched.clear()
    for message in link.poll():
        if "confirm" in message:
            ctl.ask(message["confirm"])
        elif "pick" in message:
            spot = message["pick"]
            why = ctl.start_fetch(to_job(spot["u"], spot["v"]))
            if why:
                link.send(result=why)
        elif "caregiver" in message:
            ctl.events.append(("caregiver", {"text": message["caregiver"]}))
        elif "reply" in message and ctl.task_running:
            ctl.task_done(message["reply"], now)
