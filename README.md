# sleeplog

Sleep tracking with nothing but an Android phone on the mattress and a small server. No wearable, no app store, no subscription.

- **Phone** (Termux): at night, once the screen has been off for 10 minutes, it records the accelerometer and the microphone in 15-minute chunks and uploads them. During the day it does nothing beyond a sub-second check every 15 minutes.
- **Server** (Python, stdlib + numpy/scipy): turns movement into sleep/wake with a model trained on PhysioNet's [sleep-accel](https://physionet.org/content/sleep-accel/1.0.0/) polysomnography data, detects snoring with Google's YAMNet, deletes the audio straight afterwards, and serves a one-page website.
- **Bedtime and wake time** come from Android's own record of when the phone was unlocked. Phone use at night counts as awake.

It never claims to measure deep sleep or REM: a phone can't.

## Layout

| Path | What |
|---|---|
| `phone/agent.py` | Recorder. `tick` runs from `termux-job-scheduler` every 15 min; `record` runs at night |
| `phone/test_agent.py` | Tests with Android faked out (`python3 test_agent.py`) |
| `server/app.py` | Upload endpoint, analysis worker, website |
| `server/index.html` | The website |
| `server/test_server.py`, `server/sim_test.py` | Server tests and simulated nights |
| `train_model.py`, `motion_model.json` | Motion model training script and the trained weights |

## Setup (short version)

1. **Phone:** install Termux, Termux:API and Termux:Boot from F-Droid. Grant `DUMP` and `PACKAGE_USAGE_STATS` to Termux over adb, and allow it to run in the background. On Realme/OPPO/OnePlus, also turn on **App info → Battery usage → Allow background activity**.
2. Copy `phone/agent.py` to `~/sleeplog/`, along with a `config.json` holding `{"server": "https://…", "token": "…"}`. Schedule it:
   `termux-job-scheduler --job-id 7 --period-ms 900000 --persisted true --network none --script ~/sleeplog/tick.sh`
3. **Server:** put `server/` on a box with `numpy scipy ai-edge-litert` and `ffmpeg`, add a `config.json` holding `{"token": "…", "password": "…"}` and `yamnet.tflite` (from TF Hub / Kaggle), then run `python app.py` behind HTTPS.
