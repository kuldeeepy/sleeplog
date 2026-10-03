"""Server tests: uploads, file processing and night numbers. Run: <venv>/python test_server.py (also runs sim_test.py cases)"""
import base64, gzip, http.client, json, os, socket, sys, tempfile, threading, time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import app

results = []


def check(name, cond):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name)


def fresh():
    d = tempfile.mkdtemp()
    app.DATA, app.RAW, app.EPOCHS, app.SNORE = d, os.path.join(d, "raw"), os.path.join(d, "e"), os.path.join(d, "s")
    for x in (app.RAW, app.EPOCHS, app.SNORE):
        os.makedirs(x)
    app.status["errors"] = []


# ---------- HTTP ----------
fresh()
port = socket.socket(); port.bind(("127.0.0.1", 0)); P = port.getsockname()[1]; port.close()
srv = app.ThreadingHTTPServer(("127.0.0.1", P), app.H)
threading.Thread(target=srv.serve_forever, daemon=True).start()


def req(method, path, body=b"", headers=None):
    c = http.client.HTTPConnection("127.0.0.1", P, timeout=10)
    c.request(method, path, body, headers or {})
    r = c.getresponse()
    return r.status, r.read()


tok = {"X-Token": app.CONF["token"]}
auth = {"Authorization": "Basic " + base64.b64encode(f"me:{app.CONF['password']}".encode()).decode()}
check("upload without token is refused", req("POST", "/api/upload?name=audio-1.opus", b"x")[0] == 403)
check("upload with unknown file type is refused", req("POST", "/api/upload?name=evil-1.sh", b"x", tok)[0] == 403)
check("path tricks stay inside the upload folder", req("POST", "/api/upload?name=../../audio-2.opus", b"x", tok)[0] == 200
      and os.path.exists(os.path.join(app.RAW, "audio-2.opus")))
check("good upload is saved", req("POST", "/api/upload?name=accel-3.txt.gz", b"abc", tok)[0] == 200
      and open(os.path.join(app.RAW, "accel-3.txt.gz"), "rb").read() == b"abc")
# a connection that drops mid-upload must not leave a 'complete' file
s = socket.create_connection(("127.0.0.1", P))
s.sendall(b"POST /api/upload?name=audio-4.opus HTTP/1.1\r\nHost: x\r\nX-Token: " + app.CONF["token"].encode()
          + b"\r\nContent-Length: 1000\r\n\r\nonly-part")
s.close(); time.sleep(0.5)
check("cut-off upload is thrown away", not os.path.exists(os.path.join(app.RAW, "audio-4.opus")))
check("website needs the password", req("GET", "/")[0] == 401)
check("website opens with the password", req("GET", "/", headers=auth)[0] == 200)
check("data API with the password returns JSON", json.loads(req("GET", "/api/nights", headers=auth)[1]) is not None)
srv.shutdown()

# ---------- file processing ----------
fresh()
t0 = datetime(2026, 10, 11, 1, 0, tzinfo=app.TZ).timestamp()
rng = np.random.default_rng(1)
rows = [f"{t0 + i / 45:.3f} {0.4 + rng.normal(0, .002):.4f} {0.1:.4f} {9.8 + rng.normal(0, .002):.4f}\n" for i in range(45 * 300)]
with gzip.open(os.path.join(app.RAW, f"accel-{int(t0)}.txt.gz"), "wt") as f:
    f.writelines(rows)
open(os.path.join(app.RAW, "audio-123.opus"), "wb").write(b"not really audio")
with gzip.open(os.path.join(app.RAW, "events-1.json.gz"), "wt") as f:
    json.dump([["2026-10-11 00:10:00", "KEYGUARD_HIDDEN"]], f)
with gzip.open(os.path.join(app.RAW, "events-2.json.gz"), "wt") as f:  # same event uploaded twice
    json.dump([["2026-10-11 00:10:00", "KEYGUARD_HIDDEN"]], f)
open(os.path.join(app.RAW, "audio-5.opus.part"), "wb").write(b"still uploading")
n = app.process_new()
ep = json.load(open(glob_one := os.path.join(app.EPOCHS, f"accel-{int(t0)}.txt.gz.json")))
check("5 min of motion -> about 10 thirty-second readings", 9 <= len(ep) <= 11)
check("broken sound file is set aside, not retried forever", os.path.exists(os.path.join(app.RAW, "audio-123.opus.failed")))
check("…and shows up as a note on the website", any("audio-123" in e for e in app.status["errors"]))
check("file still uploading is left alone", os.path.exists(os.path.join(app.RAW, "audio-5.opus.part")))
check("duplicate phone history is merged once", len(json.load(open(os.path.join(app.DATA, "events.json")))) == 1)
check("worker sees that something changed", n == 3)
check("…and nothing changed on a second pass", app.process_new() == 0)

# ---------- night numbers with motion + snoring ----------
fresh()
ev = []
def use(a, b):
    ev.extend([[a, "SCREEN_INTERACTIVE"], [a, "KEYGUARD_HIDDEN"], [b, "SCREEN_NON_INTERACTIVE"]])
use("2026-10-10 23:00:00", "2026-10-11 00:20:00")
use("2026-10-11 07:00:00", "2026-10-11 07:30:00")
json.dump(ev, open(os.path.join(app.DATA, "events.json"), "w"))
bed = datetime(2026, 10, 11, 0, 30, tzinfo=app.TZ).timestamp()
counts = {str(int(bed + i * 30)): float(2 + rng.random()) for i in range(13 * 60)}  # 00:30-07:00 quiet
for k in list(counts)[100::90]:
    counts[k] = 40.0  # a few big turnovers
json.dump(counts, open(os.path.join(app.EPOCHS, "a.json"), "w"))
json.dump({"start": bed + 3600, "seconds": 900, "snore": [bed + 3600 + i * 0.5 for i in range(240)]},
          open(os.path.join(app.SNORE, "s.json"), "w"))
night = app.analyse()[-1]
check("bed 00:20, up 07:00", night["bed"][11:16] == "00:20" and night["up"][11:16] == "07:00")
check("motion was used", night["motion"] and night["onset_known"])
check("turnovers counted (8 planted)", night["movements"] == 8)
check("2 min of snoring found", night["snore_min"] == 2)
check("sound coverage reported (15 min)", night["audio_min"] == 15)
check("charts line up with the timeline", len(night["motion5"]) == len(night["timeline"]) == len(night["snore5"]))
check("asleep can't exceed time in bed", night["asleep_min"] <= night["in_bed_min"])

# ---------- more awkward nights ----------
def night_of(events):
    fresh()
    json.dump(events, open(os.path.join(app.DATA, "events.json"), "w"))
    return app.analyse()

two = []
for d in (10, 11):
    two += [[f"2026-10-{d} 23:00:00", "SCREEN_INTERACTIVE"], [f"2026-10-{d} 23:00:01", "KEYGUARD_HIDDEN"], [f"2026-10-{d} 23:50:00", "SCREEN_NON_INTERACTIVE"],
            [f"2026-10-{d + 1} 07:00:00", "SCREEN_INTERACTIVE"], [f"2026-10-{d + 1} 07:00:01", "KEYGUARD_HIDDEN"], [f"2026-10-{d + 1} 08:00:00", "SCREEN_NON_INTERACTIVE"]]
check("two nights in a row -> two reports", len(night_of(two)) == 2)
late = [["2026-10-11 00:40:00", "SCREEN_INTERACTIVE"], ["2026-10-11 00:40:01", "KEYGUARD_HIDDEN"], ["2026-10-11 01:00:00", "SCREEN_NON_INTERACTIVE"],
        ["2026-10-11 08:00:00", "SCREEN_INTERACTIVE"], ["2026-10-11 08:00:01", "KEYGUARD_HIDDEN"], ["2026-10-11 08:30:00", "SCREEN_NON_INTERACTIVE"]]
n2 = night_of(late)
check("no phone use all evening, first use 00:40 -> night still found", n2 and n2[-1]["bed"][11:16] == "01:00")
nap = [["2026-10-10 18:30:00", "SCREEN_INTERACTIVE"], ["2026-10-10 18:30:01", "KEYGUARD_HIDDEN"], ["2026-10-10 19:00:00", "SCREEN_NON_INTERACTIVE"],
       ["2026-10-10 20:30:00", "SCREEN_INTERACTIVE"], ["2026-10-10 20:30:01", "KEYGUARD_HIDDEN"], ["2026-10-11 00:30:00", "SCREEN_NON_INTERACTIVE"]] + late[3:]
n3 = night_of(nap)
check("evening nap 19:00-20:30 isn't mistaken for the night", n3 and n3[-1]["bed"][11:16] == "00:30")

work = [["2026-10-11 00:00:00", "SCREEN_INTERACTIVE"], ["2026-10-11 00:00:01", "KEYGUARD_HIDDEN"], ["2026-10-11 00:30:00", "SCREEN_NON_INTERACTIVE"],
        ["2026-10-11 06:30:00", "SCREEN_INTERACTIVE"], ["2026-10-11 06:30:01", "KEYGUARD_HIDDEN"], ["2026-10-11 07:00:00", "SCREEN_NON_INTERACTIVE"],
        ["2026-10-11 12:00:00", "SCREEN_INTERACTIVE"], ["2026-10-11 12:00:01", "KEYGUARD_HIDDEN"], ["2026-10-11 12:30:00", "SCREEN_NON_INTERACTIVE"],
        ["2026-10-11 20:30:00", "SCREEN_INTERACTIVE"], ["2026-10-11 20:30:01", "KEYGUARD_HIDDEN"], ["2026-10-11 21:00:00", "SCREEN_NON_INTERACTIVE"]]
n4 = night_of(work)
check("quiet morning at work isn't counted as sleep (up 06:30)", n4 and n4[0]["up"][11:16] == "06:30")
back = [["2026-10-11 04:00:00", "SCREEN_INTERACTIVE"], ["2026-10-11 04:00:01", "KEYGUARD_HIDDEN"], ["2026-10-11 04:35:00", "SCREEN_NON_INTERACTIVE"],
        ["2026-10-11 06:17:00", "SCREEN_INTERACTIVE"], ["2026-10-11 06:17:01", "KEYGUARD_HIDDEN"], ["2026-10-11 06:38:00", "SCREEN_NON_INTERACTIVE"],
        ["2026-10-11 09:00:00", "SCREEN_INTERACTIVE"], ["2026-10-11 09:00:01", "KEYGUARD_HIDDEN"], ["2026-10-11 09:30:00", "SCREEN_NON_INTERACTIVE"]]
n5 = night_of(back)
check("woke at 6:17, back to sleep till 9:00 -> one night 04:35-09:00", n5 and n5[0]["bed"][11:16] == "04:35" and n5[0]["up"][11:16] == "09:00")

# sensor gap: 10 min with no readings must not become 'still' epochs
fresh()
rows = [f"{t0 + i / 45:.3f} 0.4 0.1 9.8\n" for i in range(45 * 120)] + [f"{t0 + 720 + i / 45:.3f} 0.4 0.1 9.8\n" for i in range(45 * 120)]
with gzip.open(os.path.join(app.RAW, f"accel-{int(t0)}.txt.gz"), "wt") as f:
    f.writelines(rows)
app.process_new()
ep = json.load(open(os.path.join(app.EPOCHS, f"accel-{int(t0)}.txt.gz.json")))
check("10-min sensor gap leaves a hole, not fake stillness", 7 <= len(ep) <= 9)

print(f"\n{sum(results)}/{len(results)} passed")
os.system(f"{sys.executable} {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim_test.py')}")
sys.exit(0 if all(results) else 1)
