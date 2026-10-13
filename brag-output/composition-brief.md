# Hyperframes Composition Brief: ZRA Railway Detection

## Objective
Create a short, polished product film for the ZRA Railway Detection app.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape, 1920x1080
- Duration: 20.0s

## Source Material
- Project root: `/home/user/ZRALabs2026`
- Primary files read: `app.py`, `.streamlit/config.toml`, `style.css`, `result.py`, `whitepaper.pdf`, `model/README.md`, `demo.mp4`
- Product name: ZRA Railway Detection (page title); "Railway Object Detection" (app heading)
- Tagline: "Upload railway-related photos, and the model marks up the assets it recognises, with a confidence score for each."
- Key UI moment: `demo.mp4` screen recording (654x388, 9.4s) of the upload area and the "What it looks for" legend
- Copy that must appear verbatim:
  - "Upload railway-related photos, and the model marks up the assets it recognises, with a confidence score for each."
  - Class descriptions from `CLASSES` in `app.py`
  - "70.7%" top-1 accuracy, "16.7%" chance baseline, "615 ms" and "2204 ms" scan times (whitepaper, section I)

## Creative Direction
- Tone preset: polished
- Creative direction: quiet, confident product film for an engineering tool
- Interpretation: slow holds, few words, numbers carry the weight, no effects beyond fades and count-ups
- Angle: one photo, six railway assets, two numbers
- Hook: "One photo." / "Six railway assets." (0.0-3.0s)
- Outro / punchline: "Point the camera at the track." (16.9-20.0s)
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Unrelated visual redesign

## Visual Identity
- Background: #0B1120
- Text: #E2E8F0
- Accent: #14B8A6 (teal, primary); link #2DD4BF
- Surface: #151E2E (cards), border #25324A
- Class colours: Train #F87171, Track #60A5FA, Signal #FACC15, Platform #4ADE80, Crossing Gate #FB923C, Overhead Wire #C084FC
- Display font: Bitcount Prop Single (project font, Google Fonts), fallback `ui-monospace, monospace`
- Body font: JetBrains Mono (project code font, Google Fonts) for captions, fallback `ui-sans-serif, system-ui`
- Visual references from the project: the logo (`assets/logo.png`), the rail motif (two-line track in #25324A), `demo.mp4`

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. Hook — 3.0s — "One photo. Six railway assets."
2. Reveal — 4.0s — logo, title, tagline, demo recording
3. The six classes — 5.65s — six badge cards, one per beat pair
4. The numbers — 4.21s — 70.7% and 615 ms count-up
5. Outro — 3.14s — logo, "Point the camera at the track."

## Audio
- Audio role: warm bed, sparse accents
- Audio arc: bed in at 0.0s, steady through the cards, fades out 18.0-20.0s
- Music: `happy-beats-business-moves-vol-9-by-ende-dot-app.mp3`, copied to `composition/assets/music/`
- Music treatment: bed level 0.5; fade out over 18.0-20.0s
- Music cue guidance: bundled preset at `skills/brag/assets/music/cues/happy-beats-business-moves-vol-9-by-ende-dot-app.music-cues.md`. Strong cues 4.23s and 12.65s are locked. Beat grid 7.40, 8.44, 9.50, 10.54, 11.60, 12.12 for the cards.
- Audio-reactive treatment: none (hyperframes-creative extraction helper not loaded this run)
- Audio-coupled moments:
  - Scene 2 — UI click on title at 4.23s
  - Scene 3 — card arrivals on beat grid, switch SFX on cards 1, 3, 5
  - Scene 4 — count-up snapped to the 12.65s strong cue
- SFX selection guidance: `sfx/ui/click1.ogg` for the title, `sfx/ui/switch1.ogg` for the cards. Keep low.
- Exact SFX choice: as above
- Audio files: copied into `composition/assets/`

## Hyperframes Instructions
Composition written against the Hyperframes docs (`npx hyperframes docs compositions|data-attributes|gsap`). Run `npx hyperframes check` in `composition/` before render. Render to `brag-output/brag.mp4`.
