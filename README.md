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

## One label table, keybinds, and how inputs are sent
* **Labels:** every label from corrections, auto-label, media analysis and recording detections is a row in the
  **`entities`** collection (same fields: `label`, `bbox`, `confidence`, `source` = `detector` / `correction` /
  `auto_label` / `media`, `profile_id`, `media_id`/`session_id`, `frame_index`, `review_status`). Filter by source in the
  Database tab. Older rows are normalised automatically at start-up. The Media collection only keeps the summary.
* **Keybinds tab:** the game keybinds *are* the profile's input mapping – add, rename, rebind, delete, un-ignore,
  **Detect key** (press the key) and **Test key** from the GUI. Purposes are stored in the `keybinds` collection.
  Duplicate keys / unknown key names are refused; the demo comparison uses the same table.
* **Input output (Settings → Input output):**
  * `window_message` (default) – inputs are posted **directly to the game window** (WM_KEYDOWN/UP, mouse messages to its
    innermost child window). No system keyboard/mouse output, no focus needed, your real keyboard is not touched.
    Works for games that read window messages (many SDL/GLFW/Win32/Chromium-based 2D games). Games that read
    DirectInput / RawInput / GetAsyncKeyState ignore it, and a game running as administrator ignores a normal app
    (run both elevated). Gamepad/touch can't be sent this way.
  * `os_input` – the previous method (system-level keyboard/mouse, game must be focused; virtual gamepad via vgamepad).
  Use **Keybinds → Test key** to see whether your game reacts to the mode you picked.

## Auto-label, elements, keybinds, media
* **Auto-label tab** – *Scan live capture*: finds recurring unlabelled elements (edges + motion), groups look-alikes
  and suggests a label (nearest known template, otherwise a hint such as `ui` / `moving_entity`). Tick the groups, fix the
  label (type a new name to create a new element) and **Apply** → template images + element entries are saved.
  Nothing is stored until you apply. Accuracy depends on the game: it is assisted labelling, not magic.
* **Elements tab** – your element catalogue (label, display name, category, on/off, notes, template images).
  Elements are added to the profile's detector labels automatically; old training examples are re-mapped, so they stay usable.
* **Database tab** – browse every collection (search, edit as JSON, add, delete, export). Direct edits bypass validation.
* **Keybind comparison** – after each recording (and before a replay, or via *Profile → Compare keybinds with a demo…*)
  the agent compares the keys/buttons used in the demo with the profile's keybinds. For every unmapped key it asks
  for the **purpose** (with a hint from what happened on screen after the key) and lets you *add to profile*,
  *ignore (never replay)* or *ask later*. It also lists mapped keys that the demo never used.
* **Media tab** – open a GIF, short video (first 30 s, up to 48 sampled frames) or image: known elements are detected
  and tracked, text is read (EN / zh-TW / ja-JP), CAPTCHA-like content is flagged, class is inferred, motion and scene
  changes are measured, and unknown recurring elements get label suggestions (same review/apply flow). Scrub/play the
  result, export an annotated GIF/MP4 or a JSON report; the analysis is stored in the `media` collection.
  Templates created from media only match a game window of the same pixel size.

## GPU (AMD) – what runs where
No CUDA needed. Settings → *GPU check…* (or Help menu) tells you what is active on your PC.
| Work | GPU path | Setup |
|---|---|---|
| Screen capture | DXGI Desktop Duplication via **dxcam** (`capture_backend` = auto/dxcam) | installed by start.bat (optional extras) |
| Image processing (template matching, resize, gray) | OpenCV **OpenCL** (`cv2.UMat`) | latest AMD Adrenalin driver |
| YOLO detection | **ONNX Runtime + DirectML** – set the profile's `detector.yolo_model` to an `.onnx` file | `onnxruntime-directml`; export with `yolo export model=best.pt format=onnx` |
| Video recording | AMD hardware encoder **h264_amf** through ffmpeg (`record_encoder` = auto/amf) | ffmpeg build with AMF on PATH (e.g. gyan.dev "full") |

Every GPU path falls back to the CPU automatically (and logs why). Honest limits: frames still have to be copied to
system memory for Python/OpenCV, OCR (Tesseract), the kNN policy and the Python loop stay on the CPU, and DXGI
capture needs the game visible (borderless window; exclusive fullscreen shows black). On a laptop with an AMD iGPU +
dGPU, set *GPU adapter index* if the wrong adapter is used.

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
| elements | your element catalogue (label, category, notes, enabled) |
| keybinds | answers to "what is this key for?" (purpose, action name, mapped/ignored) |
| media | analyses of GIFs / videos / images (summary, per-frame detections, suggestions) |
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
app/logbus.py      logging bus              app/gui/   PySide6 UI (main_window, panels, widgets)
app/autolabel.py   region proposal + clustering   app/media.py  GIF/video/image analysis
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
