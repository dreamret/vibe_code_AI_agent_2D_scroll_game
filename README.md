# Local AI Player-Action Simulator

Local desktop app (Python + PySide6) that observes a 2D side-scroller, records your
play, learns from reviewed examples and – only with your approval – replays or
proposes actions. Everything (video, frames, database, logs) stays on this PC.

## Start
Double-click **start.bat**. First run creates `.venv`, installs `requirements.txt`
(+ optional extras) and launches the app. Needs Windows + Python 3.10+ (3.11 recommended).
Run `pip freeze > requirements.lock.txt` once it works to pin exact versions
(start.bat uses the lock file when present).

## Typical workflow
1. Pick the game window (Target) and a Profile → **Capture** (red "CAPTURING" dot).
2. Switch **Learn** on, press **Record**, play normally. Inputs are stored only while
   the game window is focused. Press Record again to save the session.
3. *Corrections tab*: tick **Freeze preview**, draw a box around a missed object, give a label
   → saved as an entity correction **and** a detector template (data/templates/<profile>/<label>/).
   Put CAPTCHA screenshots/widgets into `.../_captcha/` to detect them visually.
4. *Learning tab*: review the examples (Approve / Reject / Delete) → **Train candidate model**
   (evaluated on a held-out session) → **Deploy selected version**. Nothing changes the active policy
   until you deploy.
5. **Assist / Replay ▾** → *Enable input execution* (human switch) →
   *Replay a recorded session* or *Assist mode* (every proposed action needs your click).
6. **Emergency stop: F12** (global hotkey, configurable) or the red **STOP** button.

## Languages (English, 繁體中文 zh-TW, 日本語 ja-JP)
OCR reads game text in all three. Install **Tesseract** (e.g. the UB-Mannheim Windows build) and tick the
*Traditional Chinese* (`chi_tra`) and *Japanese* (`jpn`) language data, then enable OCR in Settings
(`ocr_languages` = `eng+chi_tra+jpn`). The app warns in the log if a language pack is missing.
Recognised text is used for CAPTCHA keywords (EN/zh-TW/ja-JP built in, more via profile
`detector.captcha_keywords`), class signatures (e.g. `"劍士": ["劍士","斬擊"]`) and a script-based
text-language indicator in the Scene tab. Note: Traditional vs Simplified Chinese is reported as `zh`.
The window/menu text of the app itself is still English.

## Safety behaviour (from the proposal)
* No input is sent unless the human switch is ON, the game window is focused, the control is allowed
  by the profile and the rate limit is respected.
* Automation **pauses** on CAPTCHA, lost focus, capture loss, unknown scene, emergency stop; you press **Resume**.
* CAPTCHAs are only *detected* (OCR keywords via Tesseract, or `_captcha` templates). They are never solved or bypassed.
* Class inference returns `Unknown` unless enough evidence is visible.
* Respect each game's ToS / anti-cheat – injection may be forbidden.

## Database (local NoSQL)
Default: **TinyDB** (`data/db/sim.json`, zero setup). Recommended for big datasets: **MongoDB Community**
on localhost → Settings tab: backend `mongodb`, URI `mongodb://localhost:27017`, restart. If MongoDB
isn't reachable the app falls back to TinyDB and logs a warning. Indexes are created automatically.

| collection | content |
|---|---|
| profiles | game/window rules, input mapping, detector config, class signatures, constraints, model version |
| sessions | human recordings, replays and assist sessions (app, times, profile, capture settings, status, stats) |
| frames | sampled frames: timestamp, image path + sha1, scene state, detections, class/CAPTCHA result |
| entities | detections per frame (label, bbox, confidence, track id) + user corrections |
| actions | recorded human inputs and executed replay/assist inputs (+ `execution_result`) |
| training_examples | state features → action → outcome, context, review status (pending/approved/rejected) |
| models | policy versions, training metadata, evaluation metrics, file path, status |
| logs | timestamped structured events (level, component, event id, message, context) |

Videos / JPEGs / model files stay on disk (`data/recordings`, `data/models`); the DB stores paths and hashes.
*Data menu*: backup, export to JSON, delete a session (DB + files), purge old logs (retention in Settings).

## Project layout
```
main.py            entry point              start.bat   launcher
app/controller.py  glue / app logic         app/db.py   TinyDB+MongoDB layer
app/capture.py     capture + recording      app/perception.py  detection/OCR/CAPTCHA/class
app/input_recorder.py  keyboard/mouse/gamepad   app/action_router.py  adapters + safety
app/learning.py    kNN policy, training, evaluation   app/profiles.py  profile manager
app/logbus.py      logging bus              app/gui/   PySide6 UI
tests/test_core.py offline tests (python -m unittest discover tests -v)
```

## Known limits (honest list)
* No claim of 100 % accuracy in every game; see per-model metrics in the Learning tab.
* Capture = visible window client area (`mss`); exclusive-fullscreen games show black → use borderless mode.
  The backend is isolated in `CaptureService._region/_loop` if you want Windows Graphics Capture/DXGI.
* Detection is template matching out of the box; for real games add a YOLO model (profile `detector.yolo_model`,
  `pip install ultralytics`) or more templates.
* Gamepad *recording* needs `pygame`; gamepad *replay* needs `vgamepad` + ViGEmBus. Touch injection is not available.
* Window listing, focus checks and input injection are Windows-only.
