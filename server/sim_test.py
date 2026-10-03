"""Replays made-up nights through app.analyse() to check the night logic. Run: python sim_test.py"""
import json, os, sys, tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app

T0 = datetime(2026, 10, 10)  # an evening; nights start here


def at(day, hhmm):  # day 0 = the evening of T0, day 1 = next morning
    h, m = map(int, hhmm.split(":"))
    return (T0 + timedelta(days=day, hours=h, minutes=m)).strftime("%Y-%m-%d %H:%M:%S")


def use(day, start, end):  # an unlocked phone session
    return [(at(day, start), "SCREEN_INTERACTIVE"), (at(day, start), "KEYGUARD_HIDDEN"),
            (at(day, start), "ACTIVITY_RESUMED"), (at(day, end), "SCREEN_NON_INTERACTIVE")]


def glance(day, t):  # screen lights up (notification) but never unlocked
    return [(at(day, t), "SCREEN_INTERACTIVE"), (at(day, t), "SCREEN_NON_INTERACTIVE")]


def run(events):
    d = tempfile.mkdtemp()
    app.DATA, app.EPOCHS, app.SNORE = d, os.path.join(d, "e"), os.path.join(d, "s")
    os.makedirs(app.EPOCHS); os.makedirs(app.SNORE)
    json.dump(sorted(set(events)), open(os.path.join(d, "events.json"), "w"))
    return app.analyse()


evening = use(0, "21:00", "23:50")
cases = {
    "normal night 00:14-06:25": (evening + use(1, "00:00", "00:14") + use(1, "06:25", "07:30"),
                                 dict(bed="00:14", up="06:25", checks=0)),
    "10-min phone check at 3am": (evening + use(1, "00:00", "00:14") + use(1, "03:00", "03:10") + use(1, "06:25", "07:30"),
                                  dict(bed="00:14", up="06:25", checks=1)),
    "40-min phone use at 3am": (evening + use(1, "00:00", "00:14") + use(1, "03:00", "03:40") + use(1, "06:25", "07:30"),
                                dict(bed="00:14", up="06:25", checks=1)),
    "notification glances at night": (evening + use(1, "00:00", "00:14") + glance(1, "02:00") + glance(1, "04:30") + use(1, "06:25", "07:30"),
                                      dict(bed="00:14", up="06:25", checks=0)),
    "very late: 05:30-13:30": (evening + use(1, "00:00", "05:30") + use(1, "13:30", "14:30"),
                               dict(bed="05:30", up="13:30", checks=0)),
    "sleeps past 2pm: 06:00-14:40": (evening + use(1, "00:00", "06:00") + use(1, "14:40", "15:30"),
                                     dict(bed="06:00", up="14:40", checks=0)),
    "phone died, no morning data yet": (evening + use(1, "00:00", "00:14"), None),
    "early sleeper 22:00-06:00": (use(0, "20:00", "22:00") + use(1, "06:00", "07:00"),
                                  dict(bed="22:00", up="06:00", checks=0)),
}

fails = 0
for name, (ev, want) in cases.items():
    nights = run(ev)
    got = None
    if nights:
        n = nights[-1]
        got = dict(bed=n["bed"][11:16], up=n["up"][11:16], checks=n["phone_checks"])
    ok = got == want
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} got {got}" + ("" if ok else f"  want {want}"))
print(f"\n{len(cases) - fails}/{len(cases)} passed")
