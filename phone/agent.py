"""Phone side of sleeplog (runs in Termux). Stdlib only.

  agent.py tick     # run by Android's job scheduler every 15 min; exits in <1 s by day
  agent.py record   # started by tick at night; records motion + sound while the screen stays off
"""
import json, os, re, subprocess, sys, time, gzip, glob, threading
from datetime import datetime

HOME = os.path.expanduser("~/sleeplog")
OUT = os.path.join(HOME, "outbox")          # finished chunks waiting for upload
LOG = os.path.join(HOME, "agent.log")
PIDF = os.path.join(HOME, "record.pid")
CONF = json.load(open(os.path.join(HOME, "config.json")))  # {"server": "...", "token": "..."}
SENSOR = "icm4x6xx Accelerometer Non-wakeup"
NIGHT = (21, 13)          # recording allowed from 21:00 to 13:00
OFF_BEFORE_START = 600    # screen off this long before recording starts (s)
ON_TO_STOP = 300          # screen on this long in a row stops recording (s)
CHUNK = 900               # seconds per uploaded chunk
os.environ["PATH"] += ":/system/bin"


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
    if os.path.getsize(LOG) > 200_000:  # keep the log small
        os.replace(LOG, LOG + ".old")


def api(*cmd, timeout=15):
    """Run a termux-* command without ever letting it hang the recorder."""
    try:
        subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"{cmd[0]} timed out")


def night_now():
    h = datetime.now().hour
    return h >= NIGHT[0] or h < NIGHT[1]


def screen_on():
    out = subprocess.run(["dumpsys", "power"], capture_output=True, text=True, timeout=20).stdout
    return "mWakefulness=Awake" in out


def recording():
    try:
        cmd = open(f"/proc/{int(open(PIDF).read())}/cmdline").read()
        return "agent.py" in cmd and "record" in cmd
    except Exception:
        return False


def night_key():
    """The evening a night belongs to (a 3 am recording belongs to the evening before)."""
    return datetime.fromtimestamp(time.time() - 13 * 3600).strftime("%Y%m%d")


def last_screen_off_age():
    """Seconds since the screen last turned off, from Android's usage history (None if unknown)."""
    out = subprocess.run(["dumpsys", "usagestats"], capture_output=True, text=True, timeout=60).stdout
    offs = re.findall(r'time="([^"]+)" type=SCREEN_NON_INTERACTIVE', out.split("In-memory daily stats")[0])
    if not offs:
        return None
    return time.time() - datetime.strptime(max(offs), "%Y-%m-%d %H:%M:%S").timestamp()


def upload_pending():
    import fcntl
    lock = open(os.path.join(HOME, "upload.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return  # another upload is already running
    fails = 0
    for path in sorted(glob.glob(os.path.join(OUT, "*"))):
        try:
            r = subprocess.run(["curl", "-sf", "-m", "300", "-H", f"X-Token: {CONF['token']}",
                                "--data-binary", f"@{path}", f"{CONF['server']}/api/upload?name={os.path.basename(path)}"],
                               capture_output=True, timeout=330)
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok, r = False, None
        if ok:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            fails = 0
            continue
        log(f"upload failed {os.path.basename(path)} (curl {r.returncode if r else 'timeout'}), will retry")
        fails += 1
        if fails >= 3:  # probably no network; try again on the next tick
            return


def send_events():
    """Phone-use events (unlocks, app opens, screen on/off) for the last 24 h."""
    out = subprocess.run(["dumpsys", "usagestats"], capture_output=True, text=True, timeout=60).stdout
    keep = re.findall(r'time="([^"]+)" type=(SCREEN_INTERACTIVE|SCREEN_NON_INTERACTIVE|KEYGUARD_HIDDEN|KEYGUARD_SHOWN|ACTIVITY_RESUMED|USER_INTERACTION)',
                      out.split("In-memory daily stats")[0])
    path = os.path.join(OUT, f"events-{int(time.time())}.json.gz")
    with gzip.open(path, "wt") as f:
        json.dump(keep, f)


def screen_on_minutes(last_min):
    """Minutes the screen was on during the last `last_min` minutes."""
    out = subprocess.run(["dumpsys", "usagestats"], capture_output=True, text=True, timeout=60).stdout
    ev = re.findall(r'time="([^"]+)" type=(SCREEN_INTERACTIVE|SCREEN_NON_INTERACTIVE)', out.split("In-memory daily stats")[0])
    start, now = time.time() - last_min * 60, time.time()
    total, on_at = 0.0, None
    for t, k in sorted(ev):
        ts = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp()
        if k == "SCREEN_INTERACTIVE":
            on_at = ts
        elif on_at is not None:
            total += max(0, ts - max(on_at, start)) if ts > start else 0
            on_at = None
    if on_at is not None:
        total += now - max(on_at, start)
    return total / 60


def tick():
    os.makedirs(OUT, exist_ok=True)
    up_flag = os.path.join(HOME, f"up-{datetime.now():%Y%m%d}")
    # morning: 30+ of the last 40 min on the phone = you've started your day, no more recording until 21:00
    slept = os.path.exists(os.path.join(HOME, f"slept-{night_key()}"))
    if slept and 4 <= datetime.now().hour < NIGHT[1] and not os.path.exists(up_flag) and screen_on_minutes(40) >= 30:
        open(up_flag, "w").close()
        send_events()
        for old in sorted(glob.glob(os.path.join(HOME, "up-*")))[:-3] + sorted(glob.glob(os.path.join(HOME, "slept-*")))[:-3]:
            os.remove(old)
    if night_now() and not recording() and not screen_on() and not (datetime.now().hour < NIGHT[1] and os.path.exists(up_flag)):
        age = last_screen_off_age()
        need = 1800 if slept and 5 <= datetime.now().hour < NIGHT[1] else OFF_BEFORE_START
        if age is not None and age >= need:
            subprocess.Popen(["nohup", sys.executable, __file__, "record"], stdout=subprocess.DEVNULL,
                             stderr=open(LOG, "a"), start_new_session=True)
            log("recording started")
    sent = os.path.join(HOME, "events.sent")
    if not recording() and (not os.path.exists(sent) or time.time() - os.path.getmtime(sent) >= 3 * 3600):
        send_events()  # Android keeps only 24 h of history; the server merges repeats
        open(sent, "w").close()
        os.utime(sent, (time.time(), time.time()))
    if not recording():
        for f in glob.glob(os.path.join(HOME, "audio-*.opus")):
            if os.path.getsize(f) == 0:
                os.remove(f)
            else:
                os.replace(f, os.path.join(OUT, os.path.basename(f)))
    if os.listdir(OUT) and not recording():
        upload_pending()


def record():
    import signal
    def stop(*_):  # any stop signal ends the night cleanly (releases the wake lock, saves the last files)
        raise SystemExit
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, stop)
    open(PIDF, "w").write(str(os.getpid()))
    api("termux-wake-lock")
    api("termux-api-start")
    buf, on_since, chunk_start = [], None, time.time()
    sensor, mic = None, None
    last_reading = [time.time()]
    mic_seen = [0, time.time()]  # last size of the sound file, and when it last grew
    mic_fails, mic_retry_at = [], [0]

    def read_sensor(proc):  # stream of pretty-printed JSON objects, one per reading
        obj = ""
        for line in proc.stdout:
            obj += line
            if line.startswith("}"):
                try:
                    v = json.loads(obj).get(SENSOR, {}).get("values")
                    if v:
                        buf.append(f"{time.time():.3f} {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
                        last_reading[0] = time.time()
                except ValueError:
                    pass
                obj = ""

    def start_sensor():  # (re)start the accelerometer stream
        nonlocal sensor
        if sensor:
            sensor.terminate()
            try:
                sensor.wait(timeout=3)
            except subprocess.TimeoutExpired:
                sensor.kill()
            api("termux-sensor", "-c")
            time.sleep(1)
        sensor = subprocess.Popen(["termux-sensor", "-s", SENSOR, "-d", "20"], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=read_sensor, args=(sensor,), daemon=True).start()
        last_reading[0] = time.time()

    def stop_mic():  # finish the current sound file and queue it for upload
        api("termux-microphone-record", "-q")
        if mic and os.path.exists(mic) and os.path.getsize(mic) > 0:
            os.replace(mic, os.path.join(OUT, os.path.basename(mic)))

    def start_mic():
        nonlocal mic
        mic = os.path.join(HOME, f"audio-{int(time.time())}.opus")
        api("termux-microphone-record", "-f", mic, "-e", "opus", "-r", "16000", "-c", "1", "-b", "16", "-l", str(CHUNK + 120))
        mic_seen[:] = [0, time.time()]

    def save_motion():
        rows = list(buf)
        del buf[:len(rows)]
        if rows:
            with gzip.open(os.path.join(OUT, f"accel-{int(rows[0].split()[0].split('.')[0])}.txt.gz"), "wt") as f:
                f.writelines(rows)

    try:
        start_mic()
        time.sleep(2)
        start_sensor()
        last_check, started, errors = 0, time.time(), 0
        while True:
          try:
            now = time.time()
            if now - started >= 2 * 3600:  # mark that you really slept tonight
                open(os.path.join(HOME, f"slept-{night_key()}"), "a").close()
            # watchdogs: Android sometimes kills Termux:API, which silently stops the sensor or the microphone
            if now - last_reading[0] > 5:
                log("motion sensor went quiet, restarting it")
                api("termux-api-start")
                start_sensor()
            size = os.path.getsize(mic) if mic and os.path.exists(mic) else 0
            if size > mic_seen[0]:
                mic_seen[:] = [size, now]
            elif now - mic_seen[1] > 5 and now >= mic_retry_at[0]:
                log("microphone stopped, restarting it")
                stop_mic()
                api("termux-api-start")
                start_mic()
                # if it keeps failing (mic busy, e.g. a call), back off instead of hammering it
                mic_fails.append(now)
                del mic_fails[:-3]
                mic_retry_at[0] = now + (120 if len(mic_fails) == 3 and now - mic_fails[0] < 120 else 0)
            if now - last_check >= 20:
                last_check = now
                on_since = (on_since or now) if screen_on() else None
                if not night_now() or (on_since and now - on_since >= ON_TO_STOP):
                    break
            if now - chunk_start >= CHUNK:
                save_motion()
                stop_mic()
                start_mic()
                time.sleep(2)
                start_sensor()
                chunk_start = time.time()
            time.sleep(1)
          except Exception as e:  # log it and keep recording; give up only if it keeps happening
            errors += 1
            log(f"recording hiccup: {e!r}")
            if errors > 50:
                raise
            time.sleep(5)
    except Exception as e:
        log(f"recording error: {e!r}")
    finally:
        save_motion()
        stop_mic()
        if sensor:
            sensor.terminate()
        api("termux-sensor", "-c")
        api("termux-wake-unlock")
        os.remove(PIDF)
        log("recording stopped")
        if 4 <= datetime.now().hour < NIGHT[1]:
            send_events()  # morning: let the server build last night's report right away
        upload_pending()


if __name__ == "__main__":
    try:
        {"tick": tick, "record": record}[sys.argv[1]]()
    except Exception as e:  # never surface errors on the phone; keep them in the log
        log(f"{sys.argv[1:]} failed: {e!r}")
