# Interaction-Bench Taxonomy 2.0

This taxonomy separates **behavior setting** (Axis A) from **participation framework** (Axis B). The data-source column below lists only real-data sources we intend to incorporate from the existing-work discussion. Synthetic sources and QA-only tasks are intentionally excluded from these seed mappings.

Real-data sources retained after audit:

- **WearVox** — use only `task == "non-assistant-directed"` side-talk rejection rows. Do **not** use WearVox QA (`closed_book`, `grounding`), tool-calling, or translation rows for this taxonomy seed set.
- **HumDial-FDBench** — use the real full-duplex test folders under `HumDial-FDBench/test`; folder names provide participation/turn-taking labels, while behavior-setting labels are weaker and should be audited per utterance.

## Axis A — Behavior-setting scene classes

| Axis A class | Description | Real-data source column |
| --- | --- | --- |
| Domestic / Household | Private home life: cooking, cleaning, alarms, family requests; TV/background speech; delivery interruption. | **HumDial-FDBench** utterance content sometimes includes home/family/shopping-list examples, but folder labels do not guarantee this class; use per-row transcript audit. **WearVox** may include indoor side-talk here only when metadata/transcript supports a private household scene. |
| Work / Professional | Office, meetings, calls, desk work, factory/site work, professional coordination. | **WearVox non-assistant-directed** rows with `audio_metadata.environment = Indoors` and office-like metadata such as `_noise_type = Quiet Office`. **HumDial-FDBench** is useful for meeting-like interaction dynamics, but Axis A should remain inferred unless transcript context is explicitly professional. |
| Mobility / Transport | Driving, riding, walking navigation, transit, biking, airport/train-station movement. | `synthetic` until a row-level audit finds explicit transport examples in retained real data. Do not infer from wearable device alone. |
| Public / Civic Space | Street, cafe, store, library, park, government/public-service spaces, bystanders, strangers, public privacy constraints. | **WearVox non-assistant-directed** rows with `audio_metadata.environment = Outdoors` or public-area noise metadata; these should map here rather than to a generic “wearable” class. HumDial side-conversation folders can support this only when transcript context is public/civic. |
| Education / Learning | Classroom, tutoring, studying, lecture, children learning, language practice. | `synthetic` by default; only override after row-level HumDial/WearVox transcript audit finds explicit education/learning examples. |
| Commerce / Service | Shopping, restaurants, hotels, customer support, ticketing, banking-style interactions. | **HumDial-FDBench** and **WearVox non-assistant-directed** only after per-row transcript audit; `synthetic` otherwise. |
| Healthcare / Caregiving / Accessibility | Medical, elder care, disability assistance, caregiver-mediated interaction, medication reminders, emergency-adjacent care. | **HumDial-FDBench** and **WearVox non-assistant-directed** only after per-row transcript audit; `synthetic` otherwise. |
| Leisure / Media / Social | Music, games, entertainment, social hangouts, party/family recreation. | **WearVox non-assistant-directed** rows with music/social noise metadata can seed this class; HumDial rows can seed it only when transcript content is leisure/social. |
| Safety / Emergency / Security | Alarms, threats, accidents, urgent help, suspicious events, safety-critical commands. | `synthetic` until retained real-data rows are audited for explicit safety/emergency/security scenes. |
| Personal Admin / Communication | Messages, calls, calendar, reminders, search, email, notes — when not clearly embedded in another setting. | HumDial direct-user folders and some WearVox non-assistant-directed rows may seed this after transcript/metadata audit; `synthetic` otherwise. |

## Axis B — Participation framework

| Axis B class | Meaning | Real-data source column |
| --- | --- | --- |
| `single_user_addressed` | The primary user is addressing the assistant. | **HumDial-FDBench** `ask`, `repeat`, `deny`, and `shift` folders. These are assistant-directed user turns; `shift` is a topic shift, not a new participant join. |
| `ratified_multi_party` | Multiple known participants are legitimate parties to the assistant interaction. | `synthetic` until real multi-party ratification examples are identified by row-level audit. |
| `new_participant_joins` | A third party explicitly addresses the assistant and should be incorporated. | `synthetic` until real new-participant-joins examples are identified by row-level audit. Do not map HumDial `shift` here unless the audio/transcript shows a new participant addressing the assistant. |
| `side_conversation` | Speech is directed to someone other than the assistant, including user-to-other or other-to-user side talk. | **WearVox** `task == "non-assistant-directed"` rows; all included WearVox rows should map here. **HumDial-FDBench** `talk_to_others`, `others_talk_to_user_before`, and `others_talk_to_user_after` also map here. |
| `bystander_speech` | Ambient or overheard speech by non-participants that is not addressed to the user or assistant. | Some WearVox non-assistant-directed rows with bystander metadata may fit, but default WearVox mapping remains `side_conversation` unless annotation/transcript shows purely ambient bystander speech. |
| `overhearer_or_public_speech` | Public announcements, media, or speech meant for a public audience rather than the assistant. | `synthetic` until retained real-data rows are audited for this participation frame. |
| `ambiguous_addressee` | It is unclear whether the assistant is being addressed. | `synthetic` until retained real-data rows are audited for this participation frame. |
| `conflicting_speakers` | Multiple simultaneous speakers issue incompatible or competing instructions. | `synthetic` until retained real-data rows are audited for conflicting-speaker cases; HumDial overlap-style folders may help after transcript/timing review. |

## Axis C — Assistant action

- `respond`
- `ignore`
- `wait`
- `clarify`
- `incorporate`

## Axis D — Acoustic / channel condition

- `clean`
- `overlap`
- `reverb`
- `continuous_noise`
- `transient_noise`
- `codec_channel`
- `egocentric_motion`
- `far_field`

## Source-to-taxonomy audit notes

| Source subset | Include? | Axis A handling | Axis B handling | Notes |
| --- | --- | --- | --- | --- |
| WearVox `non-assistant-directed` | Yes | Per row from metadata/transcript: e.g. `Indoors` + `Quiet Office` → Work / Professional; `Outdoors` → Public / Civic Space; music/social metadata → Leisure / Media / Social. | `side_conversation` by default. | This is the only WearVox task retained for the current real-data seed set. |
| WearVox `closed_book`, `grounding` | No | Excluded. | Excluded. | QA data does not target participation-frame decisions. |
| WearVox `tool_call`, `live_translation` | No | Excluded. | Excluded. | Not needed for the side-conversation taxonomy seed set. |
| HumDial-FDBench `ask` | Yes | Transcript-level audit; no reliable folder-level scene. | `single_user_addressed`. | Expected action: `respond`. |
| HumDial-FDBench `repeat` | Yes | Transcript-level audit. | `single_user_addressed`. | Expected action: `respond`. |
| HumDial-FDBench `deny` | Yes | Transcript-level audit. | `single_user_addressed`. | Expected action: `respond` or `clarify` depending on final benchmark policy; current seed script uses `respond`. |
| HumDial-FDBench `shift` | Yes | Transcript-level audit. | `single_user_addressed` unless a new speaker is verified. | Topic shift is not automatically `new_participant_joins`. |
| HumDial-FDBench `pause`, `wait`, `backchannel` | Yes | Transcript-level audit. | Usually `single_user_addressed` with turn-taking wait behavior. | Expected action: `wait`. |
| HumDial-FDBench `talk_to_others` | Yes | Transcript-level audit. | `side_conversation`. | Expected action: `ignore`. |
| HumDial-FDBench `others_talk_to_user_before`, `others_talk_to_user_after` | Yes | Transcript-level audit. | `side_conversation` by default; audit for pure bystander/public speech cases. | Expected action: `ignore`. |
