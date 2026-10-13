# Brag Plan: ZRA Railway Detection

## What is this app?
A Streamlit app that takes a railway photo and marks up the assets it finds (train, track, signal, platform, crossing gate, overhead wire), each with a confidence score. The edge-deployment whitepaper pushes it toward offline use on field hardware.

## The angle
A railway photo is a pile of signals, rails, wires and gates. The model sorts them into six classes and says how sure it is. The video is a clean, confident product film: the product, the six classes, and the two numbers that matter.

## Hook (first 2-3 seconds)
Text, large: "One photo. Six railway assets." It holds on screen; nothing spins in.

## Key moments (the middle)
- The app itself: the real screen recording from `demo.mp4` (upload area, "What it looks for" legend) inside a device frame.
- The six classes as cards, each with its colour badge and the project's own description.
- Two numbers, counted up: 70.7% top-1 accuracy (chance is 16.7%), and 615 ms for a full 1280x960 scan with MobileNetV3-Small (ResNet18 takes 2204 ms).

## Outro / punchline
Logo and wordmark. Last line: "Point the camera at the track." Then the project name holds for the final second.

## User flow worth showing
Upload a photo -> the scan runs -> results come back with a confidence score per class. Only the upload area and legend are in the source recording. The scan-and-results step is not, so the video shows the upload and the classes, and does not fake a results screen.

## Tone
- Preset: polished
- Creative direction: quiet, confident product film for an engineering tool. No jokes, no hype. The numbers do the work.

## Audio direction
- Role: warm bed, sparse accents
- Music: `happy-beats-business-moves-vol-9-by-ende-dot-app.mp3` (114.84 BPM, 1:54 long, strong cues in the planning window at 4.23s, 6.34s, 10.54s, 12.65s)
- Music treatment: starts at the top, bed level 0.5, fades out over the last 2 seconds (18.0s to 20.0s)
- Music cue guidance: strong-cue locks at 4.23s (title lands) and 12.65s (stat count-up starts). Beat grid for the card sequence: 7.40, 8.44, 9.50, 10.54, 11.60 (every other beat, because the cards are text).
- Audio-reactive treatment: none. The extraction helper ships with the hyperframes-creative skill, which this run did not load. Skipped, documented here.
- SFX posture: sparse. One UI click on the upload-area scene, one switch on the card reveals, nothing else.
- Restraint rule: no SFX on the hook or the outro. The stat count-up gets no effect sound.

## Storyboard

### Scene 1 — Hook — 0.0s to 3.0s (3.0s)
Dark slate background (#0B1120). Two lines of display type: "One photo." then "Six railway assets." Each enters in 0.4s and holds until 3.0s. Hold on the second line is 1.6s.
Sequential/interaction: none
Audio intent: silence, then the music bed comes in under the hook
Audio-coupled idea: none
Music: bed fades in from 0.0s
Transition mood: hard cut -> Scene 2

### Scene 2 — Reveal — 3.0s to 7.0s (4.0s)
Left: the project name "ZRA Railway Detection" and the logo (`logo.png`). The title lands at 4.23s (strong cue). The tagline is set below in body type, verbatim: "Upload railway-related photos, and the model marks up the assets it recognises, with a confidence score for each." Right: the `demo.mp4` screen recording in a device frame, playing at its native rate.
Sequential/interaction: none
Audio intent: a clean click when the title lands
Audio-coupled idea: UI click on the title at 4.23s
Music: bed continues
Transition mood: soft -> Scene 3

### Scene 3 — The six classes — 7.0s to 12.65s (5.65s)
Heading "Six classes." Six cards in a 3x2 grid. Each card: a badge in its class colour (Train red, Track blue, Signal yellow, Platform green, Crossing Gate orange, Overhead Wire violet) and the description from `CLASSES` in `app.py`. Cards 1-4 enter one at a time on the beat grid at 7.40, 8.44, 9.50, 10.54. Cards 5 and 6 land together at 11.60, the last beat pair before the cut at 12.65.
Sequential/interaction: yes — six cards arrive one by one, each with a switch sound on its entry.
Audio intent: quick, tidy arrivals that follow the pulse
Audio-coupled idea: card-by-card reveal on the beat grid, SFX on every other card
Music: bed, slightly louder under the cards
Transition mood: clean -> Scene 4

### Scene 4 — The numbers — 12.65s to 16.86s (4.21s)
Two stat blocks side by side. Left: "70.7%" with "top-1 accuracy, chance is 16.7%". Right: "615 ms" with "full 1280x960 scan, MobileNetV3-Small on a mid-tier laptop (ResNet18: 2204 ms)". Both count up from zero, starting at 12.65s (strong cue). Count-up runs 1.2s; both values hold to 16.86s.
Sequential/interaction: count-up, left then right, 0.2s apart
Audio intent: the bed carries it; no effect sound
Audio-coupled idea: count-up snapped to the 12.65s strong cue
Music: bed
Transition mood: hard -> Scene 5

### Scene 5 — Outro — 16.86s to 20.0s (3.14s)
Logo and "ZRA Railway Detection", centred. Line below: "Point the camera at the track." The line enters over 0.4s and holds to 20.0s. Music fades out from 18.0s to 20.0s.
Sequential/interaction: none
Audio intent: music resolves
Audio-coupled idea: none
Music: fades out from 18.0s
Transition mood: soft fade to black

**Music mood for this video:** cinematic-upbeat, quiet and steady
**Audio summary:** a music bed in from the first frame, one click when the title lands, six card arrivals on the beat grid, the bed fading out under the outro.

Duration check: 3.0 + 4.0 + 5.65 + 4.21 + 3.14 = 20.00s.
