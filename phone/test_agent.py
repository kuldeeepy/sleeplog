"""Tests for the phone agent with Android faked out. Run on the Mac: python3 test_agent.py"""
import gzip, json, os, subprocess, sys, tempfile, types
from datetime import datetime, timedelta

HOME = tempfile.mkdtemp()
os.environ["HOME"] = HOME
os.makedirs(os.path.join(HOME, "sleeplog/outbox"))
json.dump({"server": "https://x", "token": "t"}, open(os.path.join(HOME, "sleeplog/config.json"), "w"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent

S = os.path.join(HOME, "sleeplog")
fake = types.SimpleNamespace(now=None, screen=False, events=[], curl_rc={}, popen=[])


class FakeDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return fake.now


agent.datetime = FakeDT
agent.time.time = lambda: fake.now.timestamp()


def run(cmd, **kw):
    out = ""
    if cmd[:2] == ["dumpsys", "power"]:
        out = "mWakefulness=Awake" if fake.screen else "mWakefulness=Asleep"
    elif cmd[:2] == ["dumpsys", "usagestats"]:
        out = "".join(f'    time="{t:%Y-%m-%d %H:%M:%S}" type={k} package=android\n' for t, k in fake.events) + "In-memory daily stats"
    elif cmd[0] == "curl":
        name = cmd[-1].split("name=")[1]
        rc = fake.curl_rc.get(name, 0)
        return subprocess.CompletedProcess(cmd, rc, b"", b"")
    return subprocess.CompletedProcess(cmd, 0, out, "")


agent.subprocess.run = run
agent.subprocess.Popen = lambda cmd, **kw: fake.popen.append(cmd)
agent.api = lambda *a, **k: None


def reset(now, screen=False, events=()):
    for f in os.listdir(S):
        p = os.path.join(S, f)
        if os.path.isfile(p) and f != "config.json":
            os.remove(p)
    for f in os.listdir(agent.OUT):
        os.remove(os.path.join(agent.OUT, f))
    fake.now, fake.screen, fake.events, fake.curl_rc, fake.popen = now, screen, list(events), {}, []


D = datetime(2026, 10, 10)
results = []


def check(name, cond):
    results.append(cond)
    print(("PASS  " if cond else "FAIL  ") + name)


# daytime: nothing starts, phone-use history is sent once a day
reset(D.replace(hour=15), events=[(D.replace(hour=14), "SCREEN_NON_INTERACTIVE")])
agent.tick()
check("daytime tick never starts recording", fake.popen == [])
check("daytime tick sends phone history", os.path.exists(f"{S}/events.sent") and not os.listdir(agent.OUT))
fake.now += timedelta(minutes=15)
agent.tick()
check("…but not again 15 min later (no daytime work)", not any(f.startswith("events-") for f in os.listdir(agent.OUT)) and fake.popen == [])
os.utime(f"{S}/events.sent", (fake.now.timestamp() - 4 * 3600,) * 2)
fake.curl_rc = {}
sent_before = os.path.getmtime(f"{S}/events.sent")
agent.tick()
check("…and again after 3 hours (Android keeps only 24 h)", os.path.getmtime(f"{S}/events.sent") > sent_before)

# night: starts only after 10 min of screen off
reset(D.replace(hour=23), events=[(D.replace(hour=22, minute=48), "SCREEN_NON_INTERACTIVE")])
agent.tick()
check("23:00, screen off 12 min -> recording starts", len(fake.popen) == 1)
reset(D.replace(hour=23), events=[(D.replace(hour=22, minute=55), "SCREEN_NON_INTERACTIVE")])
agent.tick()
check("23:00, screen off 5 min -> waits", fake.popen == [])
reset(D.replace(hour=23), screen=True, events=[(D.replace(hour=22, minute=40), "SCREEN_NON_INTERACTIVE")])
agent.tick()
check("23:00, using the phone -> waits", fake.popen == [])

# a 4:35 am bedtime after heavy phone use must still record (no 'you're up' flag before sleeping)
use = [(D + timedelta(hours=27, minutes=40), "SCREEN_INTERACTIVE")]
reset(D + timedelta(hours=28, minutes=20), screen=True, events=use)
agent.tick()
check("4:20 am heavy use before sleeping does not set 'up' flag", not os.path.exists(f"{S}/up-20261011"))
fake.screen, fake.events = False, use + [(D + timedelta(hours=28, minutes=35), "SCREEN_NON_INTERACTIVE")]
fake.now = D + timedelta(hours=28, minutes=50)
agent.tick()
check("…and recording starts when you fall asleep at 4:35", len(fake.popen) == 1)

# morning after a real sleep: 30+ min on the phone -> 'up' flag, history sent, no more recording
reset(D + timedelta(hours=31, minutes=10), screen=True, events=[(D + timedelta(hours=30, minutes=30), "SCREEN_INTERACTIVE")])
open(f"{S}/slept-20261010", "w").close()
agent.tick()
check("7:10 am after sleeping, 40 min of use -> 'up' flag", os.path.exists(f"{S}/up-20261011"))
check("…and phone history queued/sent right away", not any(f.startswith("events-") for f in os.listdir(agent.OUT)))
fake.screen, fake.now = False, D + timedelta(hours=32)
fake.events = [(D + timedelta(hours=31, minutes=40), "SCREEN_NON_INTERACTIVE")]
agent.tick()
check("…phone down at 7:40 does not restart recording", fake.popen == [])

# stale pid file (Termux was killed, pid reused by something else)
reset(D.replace(hour=23))
open(agent.PIDF, "w").write(str(os.getpid()))  # this test process is not 'agent.py record'
check("stale pid file is not mistaken for a running recording", agent.recording() is False)

# uploads: one bad file doesn't block the rest; no network stops after 3 tries
reset(D.replace(hour=15))
for n in ("accel-1.txt.gz", "audio-2.opus", "audio-3.opus"):
    open(os.path.join(agent.OUT, n), "w").write("x")
fake.curl_rc = {"accel-1.txt.gz": 22}
agent.upload_pending()
check("a failing file doesn't block the others", sorted(os.listdir(agent.OUT)) == ["accel-1.txt.gz"])
for n in ("a-1", "a-2", "a-3", "a-4"):
    open(os.path.join(agent.OUT, n), "w").write("x")
    fake.curl_rc[n] = 7
agent.upload_pending()
check("no network: stops after 3 failures, keeps files", len(os.listdir(agent.OUT)) == 5)

# a sound clip stranded by a killed recorder is moved to the outbox and uploaded
reset(D.replace(hour=15))
open(f"{S}/audio-99.opus", "w").write("x")
agent.tick()
check("stranded sound clip is rescued and uploaded", not os.path.exists(f"{S}/audio-99.opus") and not os.listdir(agent.OUT))

# after sleeping, a short morning check + phone down 10 min must NOT restart recording; 30 min (back to sleep) does
reset(D + timedelta(hours=31), events=[(D + timedelta(hours=30, minutes=48), "SCREEN_NON_INTERACTIVE")])
open(f"{S}/slept-20261010", "w").close()
agent.tick()
check("7:00 am, slept, phone down 12 min -> no new recording", fake.popen == [])
fake.events = [(D + timedelta(hours=30, minutes=25), "SCREEN_NON_INTERACTIVE")]
agent.tick()
check("7:00 am, slept, phone down 35 min (back to sleep) -> records", len(fake.popen) == 1)

# an empty sound file left by a failed mic restart is deleted, not uploaded
reset(D.replace(hour=15))
open(f"{S}/audio-98.opus", "w").close()
agent.tick()
check("empty sound file is deleted", not os.path.exists(f"{S}/audio-98.opus") and not os.listdir(agent.OUT))

# night_key: a 3 am moment belongs to the evening before
fake.now = D + timedelta(hours=27)
check("3 am belongs to the previous evening", agent.night_key() == "20261010")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
