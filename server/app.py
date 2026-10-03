"""sleeplog server: receives the phone's uploads, works out each night, serves the website.

Listens on 127.0.0.1:8090; put it behind HTTPS (e.g. Tailscale Funnel).
"""
import base64, glob, gzip, json, os, subprocess, threading, time, traceback
from datetime import datetime, timedelta
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from zoneinfo import ZoneInfo
import numpy as np
from scipy.signal import butter, filtfilt

D = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(D, "data")
RAW, EPOCHS, SNORE = (os.path.join(DATA, x) for x in ("raw", "epochs", "snore"))
CONF = json.load(open(os.path.join(D, "config.json")))  # {"token": "...", "password": "..."}
MODEL = json.load(open(os.path.join(D, "motion_model.json")))
TZ = ZoneInfo("Asia/Kolkata")
USE = {"KEYGUARD_HIDDEN", "ACTIVITY_RESUMED", "USER_INTERACTION"}
FS, EP = 50, 30
B, A = butter(2, 0.25 / (FS / 2))
SNORE_IDX, SNORE_THR = 38, 0.2   # YAMNet "Snoring" class; threshold from the fan-noise test
status = {"errors": []}


def write_json(path, obj):
    """Write via a temp file so a crash never leaves a half-written file behind."""
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def note_error(msg):
    status["errors"] = (status["errors"] + [f"{datetime.now(TZ):%d %b %H:%M} {msg}"])[-5:]


# ---------- per-file processing ----------

def do_accel(path):
    """Raw accelerometer chunk -> activity count per 30 s epoch (same method the model was trained with)."""
    m = np.loadtxt(gzip.open(path, "rt"), ndmin=2)
    if len(m) < 20 * 40:  # under ~20 s of readings: nothing useful
        return
    t0 = np.floor(m[0, 0] / EP) * EP
    tg = np.arange(m[0, 0], m[-1, 0], 1 / FS)
    mag = np.linalg.norm(np.column_stack([np.interp(tg, m[:, 0], m[:, i]) for i in (1, 2, 3)]), axis=1)
    hp = np.abs(mag - filtfilt(B, A, mag))
    idx = ((tg - t0) // EP).astype(int)
    cnt = np.bincount(idx, hp)
    n = np.bincount(((m[:, 0] - t0) // EP).astype(int), minlength=len(cnt))[:len(cnt)] * FS / 45  # real readings, ~45/s
    out = {str(int(t0 + i * EP)): float(c) for i, (c, k) in enumerate(zip(cnt, n)) if k >= 0.5 * FS * EP}
    write_json(os.path.join(EPOCHS, os.path.basename(path) + ".json"), out)


def do_audio(path):
    """Opus chunk -> list of times (epoch s) where YAMNet hears snoring; the audio is deleted afterwards."""
    from ai_edge_litert.interpreter import Interpreter
    start = int(os.path.basename(path).split("-")[1].split(".")[0])
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, np.float32)
    yam = Interpreter(os.path.join(D, "yamnet.tflite"))
    yam.allocate_tensors()
    i_in, i_out = yam.get_input_details()[0]["index"], yam.get_output_details()[0]["index"]
    hits = []
    for i in range(0, len(x) - 15600, 8000):  # 0.975 s frames, 0.5 s hop
        yam.set_tensor(i_in, x[i:i + 15600])
        yam.invoke()
        if yam.get_tensor(i_out)[0, SNORE_IDX] > SNORE_THR:
            hits.append(start + i / 16000)
    # a snore episode needs >= 3 snore frames within 30 s (filters one-off breathing sounds)
    episodes = [t for k, t in enumerate(hits) if sum(1 for u in hits[max(0, k - 6):k + 7] if abs(u - t) <= 15) >= 3]
    write_json(os.path.join(SNORE, os.path.basename(path) + ".json"), {"start": start, "seconds": len(x) / 16000, "snore": episodes})


def do_events(path):
    ev = {tuple(e) for e in json.load(open(os.path.join(DATA, "events.json")))} if os.path.exists(os.path.join(DATA, "events.json")) else set()
    ev |= {tuple(e) for e in json.load(gzip.open(path, "rt"))}
    write_json(os.path.join(DATA, "events.json"), sorted(ev))


def process_new():
    done = 0
    for path in sorted(glob.glob(os.path.join(RAW, "*"))):
        name = os.path.basename(path)
        if name.endswith((".part", ".failed")):  # still uploading, or already given up on
            continue
        try:
            if name.startswith("accel-"):
                do_accel(path)
            elif name.startswith("audio-"):
                do_audio(path)
            elif name.startswith("events-"):
                do_events(path)
            os.remove(path)
            done += 1
        except Exception as e:
            note_error(f"could not process {name}: {e}")
            traceback.print_exc()
            try:
                os.replace(path, path + ".failed")
            except OSError:
                pass
    return done


# ---------- nightly analysis ----------

def model_sleep_prob(counts):
    x = np.log1p(np.asarray(counts, float))
    lo, mid, hi = np.percentile(x, [10, 50, 90])
    s = (x - mid) / max(hi - lo, 1e-6)
    f = [s]
    for k in MODEL["windows"]:
        pad = np.pad(s, (k // 2, k // 2), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, k)
        f += [win.mean(1), win.max(1)]
    pad = np.pad(s, (5, 5), mode="edge")
    f.append(np.lib.stride_tricks.sliding_window_view(pad, 11).std(1, ddof=1))
    f.append(np.linspace(0, 1, len(s)))
    z = np.column_stack(f) @ np.array(MODEL["coef"]) + MODEL["intercept"]
    return 1 / (1 + np.exp(-z))


def analyse():
    if not os.path.exists(os.path.join(DATA, "events.json")):
        return []
    ev = json.load(open(os.path.join(DATA, "events.json")))
    ts_of = lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()
    # phone sessions: the whole time the screen is on and unlocked, plus single app-use events
    unlocks = np.array(sorted(ts_of(t) for t, k in ev if k == "KEYGUARD_HIDDEN"))
    sess, on = [(ts_of(t), ts_of(t)) for t, k in ev if k in USE], None
    for t, k in sorted(ev, key=lambda e: ts_of(e[0])):
        if k == "SCREEN_INTERACTIVE" and on is None:
            on = ts_of(t)
        elif k == "SCREEN_NON_INTERACTIVE" and on is not None:
            if ((unlocks >= on - 3) & (unlocks <= ts_of(t))).any():  # it was unlocked, so it was really used
                sess.append((on, ts_of(t)))
            on = None
    merged = []
    for st, en in sorted(sess):  # join sessions less than 2 min apart
        if merged and st - merged[-1][1] <= 120:
            merged[-1] = (merged[-1][0], max(merged[-1][1], en))
        else:
            merged.append((st, en))
    sess = merged
    epochs = {}
    for f in glob.glob(os.path.join(EPOCHS, "*.json")):
        epochs.update({int(k): v for k, v in json.load(open(f)).items()})
    snore_t, audio_cover = [], []
    for f in glob.glob(os.path.join(SNORE, "*.json")):
        j = json.load(open(f))
        snore_t += j["snore"]
        audio_cover.append((j["start"], j["start"] + j["seconds"]))
    snore_t = np.array(sorted(snore_t))

    nights = []
    if not sess:
        return nights
    day = datetime.fromtimestamp(sess[0][0], TZ).date() - timedelta(days=1)
    while day <= datetime.fromtimestamp(sess[-1][0], TZ).date():
        lo = datetime(day.year, day.month, day.day, 18, tzinfo=TZ).timestamp()
        hi = lo + 20 * 3600
        day += timedelta(days=1)
        ss = [x for x in sess if x[1] >= lo and x[0] <= hi]
        after = [x for x in sess if x[0] > hi]
        if after and not any(x[0] > lo + 11 * 3600 for x in ss):
            ss.append(after[0])  # no phone use since 5 am: still asleep past 2 pm, the next session ends the night
        if len(ss) < 2:
            continue  # need phone use before and after the sleep
        gaps = [(x[1], y[0]) for x, y in zip(ss, ss[1:]) if x[1] < hi]
        if not gaps:
            continue
        i = max(range(len(gaps)), key=lambda k: gaps[k][1] - gaps[k][0])
        if gaps[i][1] - gaps[i][0] < 3600:
            continue
        a = b = i  # join phone-free stretches (>=30 min) across phone use of up to 60 min
        while a > 0 and gaps[a - 1][1] - gaps[a - 1][0] >= 1800 and gaps[a][0] - gaps[a - 1][1] <= 3600:
            a -= 1
        def in_bed(g):  # a later stretch is more sleep only if it ends by 11 am or the phone recorded it on the mattress
            covered = sum(1 for k in epochs if g[0] <= k < g[1]) * EP
            return g[1] <= lo + 17 * 3600 or covered >= 0.5 * (g[1] - g[0])
        while b < len(gaps) - 1 and gaps[b + 1][1] - gaps[b + 1][0] >= 1800 and gaps[b + 1][0] - gaps[b][1] <= 3600 \
                and in_bed(gaps[b + 1]):
            b += 1
        bed, up = gaps[a][0], gaps[b][1]
        phone_breaks = [(gaps[k][1], gaps[k + 1][0]) for k in range(a, b)]

        # motion inside the phone-free stretches
        keys = sorted(k for k in epochs if bed <= k < up)
        restless = []
        fell_asleep = bed
        if len(keys) >= 40:
            p = model_sleep_prob([epochs[k] for k in keys])
            wake = p < MODEL["threshold"]
            for k, w in zip(keys, wake):
                if w:
                    restless.append(k)
            run = 0
            if keys[0] - bed <= 1800:  # recording began soon after the phone was put down
                for k, w in zip(keys, wake):
                    run = 0 if w else run + 1
                    if run >= 10:
                        fell_asleep = k - 9 * EP
                        break
        restless_after = [k for k in restless if k >= fell_asleep]
        onset_known = bool(keys) and keys[0] - bed <= 1800
        counts = np.array([epochs[k] for k in keys]) if keys else np.array([])
        movements = int((counts > 5 * np.median(counts)).sum()) if len(counts) else 0  # big movements, e.g. turning over
        phone_min = sum(e - s for s, e in phone_breaks) / 60
        in_bed_min = (up - fell_asleep) / 60
        restless_min = len(restless_after) * EP / 60
        asleep_min = max(in_bed_min - phone_min - restless_min, 0)
        sn = snore_t[(snore_t >= bed) & (snore_t < up)]
        audio_min = sum(max(0, min(e, up) - max(s, bed)) for s, e in audio_cover) / 60

        # 5-minute timeline for the website
        timeline = []
        for t in np.arange(bed, up, 300):
            if any(s <= t < e for s, e in phone_breaks):
                state = "phone"
            elif t < fell_asleep:
                state = "falling"
            elif sum(1 for k in restless_after if t <= k < t + 300) >= 4:
                state = "restless"
            elif ((sn >= t) & (sn < t + 300)).sum() >= 2:
                state = "snore"
            else:
                state = "asleep"
            timeline.append(state)

        nights.append({
            "date": datetime.fromtimestamp(lo, TZ).strftime("%Y-%m-%d"),
            "bed": datetime.fromtimestamp(bed, TZ).isoformat(),
            "fell_asleep": datetime.fromtimestamp(fell_asleep, TZ).isoformat(),
            "up": datetime.fromtimestamp(up, TZ).isoformat(),
            "asleep_min": round(asleep_min),
            "in_bed_min": round((up - bed) / 60),
            "phone_checks": len(phone_breaks), "phone_min": round(phone_min),
            "restless_min": round(restless_min),
            "snore_min": round(len(sn) * 0.5 / 60),
            "motion": len(keys) >= 40, "audio_min": round(audio_min),
            "onset_known": onset_known, "movements": movements,
            "timeline": timeline,
            # per 5 min: how much you moved (0-1, relative to the night) and seconds of snoring, None = not recorded
            "motion5": [None if not any(t <= k < t + 300 for k in keys) else
                        round(float(np.mean([epochs[k] for k in keys if t <= k < t + 300])) / max(float(np.percentile(counts, 99)), 1e-6), 3)
                        for t in np.arange(bed, up, 300)] if len(keys) else [],
            "snore5": [int(((sn >= t) & (sn < t + 300)).sum() * 0.5) for t in np.arange(bed, up, 300)],
        })
    return nights


def worker():
    first = True
    while True:
        try:
            if not process_new() and not first:
                time.sleep(120)
                continue
            first = False
            nights = analyse()
            write_json(os.path.join(DATA, "nights.json"), {"nights": nights, "updated": datetime.now(TZ).isoformat(),
                                                           "errors": status["errors"], "last_upload": status.get("last_upload")})
        except Exception as e:
            note_error(f"analysis failed: {e}")
            traceback.print_exc()
        time.sleep(120)


# ---------- HTTP ----------

class H(BaseHTTPRequestHandler):
    timeout = 60
    def _authed_site(self):
        want = "Basic " + base64.b64encode(f"me:{CONF['password']}".encode()).decode()
        if self.headers.get("Authorization") == want:
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="sleep"')
        self.end_headers()
        return False

    def _send(self, code, body, ctype="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        u = urlparse(self.path)
        name = os.path.basename(parse_qs(u.query).get("name", [""])[0])
        if u.path != "/api/upload" or self.headers.get("X-Token") != CONF["token"] \
                or not name.split("-")[0] in ("accel", "audio", "events"):
            return self._send(403, b"no")
        n = int(self.headers.get("Content-Length", 0))
        if n > 200_000_000:
            return self._send(413, b"too big")
        tmp = os.path.join(RAW, name + ".part")
        with open(tmp, "wb") as f:
            left = n
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            os.remove(tmp)
            return self._send(400, b"incomplete")
        os.replace(tmp, os.path.join(RAW, name))
        status["last_upload"] = datetime.now(TZ).isoformat()
        self._send(200, b"ok")

    def do_GET(self):
        if not self._authed_site():
            return
        p = urlparse(self.path).path
        if p == "/api/nights":
            f = os.path.join(DATA, "nights.json")
            return self._send(200, open(f, "rb").read() if os.path.exists(f) else b'{"nights":[]}', "application/json")
        return self._send(200, open(os.path.join(D, "index.html"), "rb").read(), "text/html; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    for d in (RAW, EPOCHS, SNORE):
        os.makedirs(d, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", 8090), H).serve_forever()
