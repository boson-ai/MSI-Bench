# Interaction-Bench

A comprehensive benchmark and dataset for evaluating whether an audio understanding model can reliably decide **what to answer and whether to answer at all** when speech, speakers, activity setting, addressee relationships, devices, and environmental sounds create ambiguity.

## Core Research Question

Can an audio understanding model reliably decide what to answer and whether to answer at all when speech, speakers, activity setting, addressee relationships, devices, and environmental sounds create ambiguity?

The benchmark focuses on **text answers** as outputs, not transcript quality. Transcripts may be retained as metadata, but the primary evaluation target is situated assistant behavior.

## Motivation

Modern Audio Understanding Models are increasingly used as interactive assistants across homes, workplaces, vehicles, public spaces, service encounters, care settings, and personal communication workflows. Existing benchmarks cover pieces of this space, but they do not fully measure whether a model can both **understand an audio scene** and make the correct **interaction decision**: answer, incorporate a new participant, wait, ask for clarification, or stay silent.

Interaction-Bench treats "scene" as a **behavior setting**: a recurring activity pattern in a physical and social environment. This avoids overlapping categories such as "meeting," "car," and "wearable," which mix activity type, location, mobility, and device form factor.

## Scope

- **Behavior settings** across 10 mutually exclusive primary scene classes:
  - `domestic_household` — private household routines and family life
  - `work_professional` — work, meetings, professional coordination
  - `mobility_transport` — movement through the world or vehicle-mediated activity
  - `public_civic` — shared public spaces (street, park, library, government office)
  - `education_learning` — teaching, learning, tutoring, classroom, training
  - `commerce_service` — buying, selling, hospitality, customer service
  - `healthcare_caregiving_accessibility` — medical, caregiving, elder-care, accessibility
  - `leisure_media_social` — entertainment, hobbies, games, informal socializing
  - `safety_emergency_security` — urgent safety, threat, accident, alarm
  - `personal_admin_communication` — cross-setting personal organization and communication
- **Addressee and engagement detection**: addressed speech, side conversation, newly ratified participant, incomplete/interrupted turn, ambiguous/conflicting request
- **Event-triggered monitoring**: persistent listening after a user-defined condition, silence before trigger, timely response after trigger, distractor rejection
- **Acoustic perturbation testing**: reverb, overlapping speech, transient noise, continuous noise, channel degradation, egocentric motion
- **Evaluation labels and metrics** for response correctness, silence correctness, wait/clarification correctness, trigger timing, robustness degradation, hallucinated engagement, and who/when/how failure diagnosis

## Benchmark Taxonomy

### Axis A: Primary Behavior-Setting Scene Class

Each example receives exactly one primary scene class, chosen by the dominant activity frame. Device type, acoustic condition, and participation structure are separate metadata.

| Scene class | Definition | Examples |
| --- | --- | --- |
| `domestic_household` | Private household routines and family life | cooking, smart-home control, TV background, delivery interruption |
| `work_professional` | Work, meetings, professional coordination | meeting assistant, desk work, factory/site coordination |
| `mobility_transport` | Movement through the world or vehicle-mediated activity | driving, passenger use, walking navigation, transit |
| `public_civic` | Shared public spaces not primarily commercial/medical/educational | street, park, library, government office |
| `education_learning` | Teaching, learning, tutoring, studying | lecture Q&A, language practice, child tutoring |
| `commerce_service` | Buying, selling, hospitality, customer service | store, restaurant, hotel, ticketing |
| `healthcare_caregiving_accessibility` | Medical, caregiving, elder-care, accessibility | medication reminder, caregiver dialogue, symptom triage |
| `leisure_media_social` | Entertainment, hobbies, games, informal socializing | music, podcast, party, game assistant |
| `safety_emergency_security` | Urgent safety, threat, accident, alarm | fall detection, fire alarm, emergency call |
| `personal_admin_communication` | Cross-setting personal organization (residual class) | messages, calls, calendar, reminders, notes |

### Axis B: Participation Framework

| Participation frame | Expected challenge |
| --- | --- |
| `single_user_addressed` | One user directly addresses the assistant |
| `ratified_multi_party` | Multiple accepted participants in the interaction |
| `new_participant_joins` | A third party explicitly addresses the assistant |
| `side_conversation` | Speech near the assistant addressed to another human |
| `bystander_speech` | Nearby intelligible speech not part of the interaction |
| `overhearer_or_public_speech` | Public/ambient speech not directed to the assistant |
| `ambiguous_addressee` | Unclear whether the assistant is addressed |
| `conflicting_speakers` | Multiple speakers issue incompatible instructions |

### Axis C: Expected Assistant Action

| Action | Meaning | Typical failure |
| --- | --- | --- |
| `respond` | Assistant is addressed and should answer | Missed engagement or wrong answer |
| `ignore` | Speech not addressed to the assistant | False engagement / hallucinated response |
| `wait` | Turn is incomplete, interrupted, or socially premature | Premature response |
| `clarify` | Assistant should ask a clarification question | Hallucinated intent or wrong addressee |
| `incorporate` | New participant joins and addresses the assistant | Ignores participant or treats as bystander |
| `monitor` | Keep listening after user-defined trigger; do not answer yet | Premature response before trigger |
| `triggered_respond` | Respond because monitored trigger event occurred | Missed trigger, late response, wrong-trigger response |

### Axis D: Trigger and Temporal Context

- `trigger_condition` — user-defined condition that must be satisfied before response
- `trigger_event_type` — speech event, environmental sound, device signal, safety cue, or mixed
- `trigger_status` — pre-trigger, trigger-present, post-trigger, distractor-only, or no-trigger
- `response_window` — acceptable time interval after the event
- `silence_window` — interval before the event where any answer counts as premature
- `distractor_event` — similar but non-target event to test over-triggering

### Axis E: Acoustic Robustness

Each scenario can be evaluated under clean and perturbed conditions:
1. **Overlap** — partial and full speech overlap, competing side conversations
2. **Reverb** — meeting room, hallway, car cabin, kitchen, large public space
3. **Transient noise** — honks, sirens, notification sounds, door knocks
4. **Continuous noise** — wind, road noise, engine noise, crowd, TV, music, HVAC
5. **Codec/channel** — phone call, Bluetooth, compressed audio, far-field microphone
6. **Egocentric movement** — wearable motion, head turns, changing speaker distance

## Benchmark Tasks

1. **Scene-Conditioned Addressee Detection** — decide whether the assistant is addressed within a behavior setting and participation frame
2. **Situated QA** — answer only if the user or accepted participant addresses the agent
3. **Instruction Following / Rejection** — execute, reject, wait, or clarify depending on addressee and context
4. **Participation Update** — incorporate a newly ratified participant without responding to unrelated bystanders
5. **Event-Triggered Monitoring** — remain silent while monitoring, then respond within the response window when the trigger occurs
6. **Noise-Stress Evaluation** — compare task accuracy under clean vs perturbed audio

## Metrics

### Primary Metrics

- **Engagement Accuracy** — correct respond / ignore / wait / clarify / incorporate / monitor / triggered_respond decision
- **False Engagement Rate** — percentage of must-ignore/wait/monitor cases where the model responds
- **Missed Engagement Rate** — percentage of must-respond/incorporate/triggered_respond cases where the model stays silent
- **Clarification Accuracy** — percentage of ambiguous/conflicting cases where the model asks an appropriate clarification
- **Answer Accuracy** — correctness of the text answer when response is required
- **Trigger Detection Accuracy** — percentage of event-triggered cases where the model correctly distinguishes triggers from distractors
- **Temporal Response Accuracy** — percentage of triggered-response cases inside the acceptable response window
- **Robustness Drop** — performance degradation from clean to perturbed versions
- **Perturbation Sensitivity Curve** — score as a function of SNR, overlap ratio, RT60/reverb level, and codec quality

### Who/When/How Diagnostics

- **Who failure** — model assigns a request, event, or answer to the wrong speaker or participant role
- **When failure** — model responds too early, too late, misses the response window, or fails to wait
- **How failure** — correct engagement timing but incorrect, inappropriate, or under-informative answer (optional)

### Required Report Slices

- Primary behavior-setting scene class, participation frame, expected action
- Trigger status and event type
- Acoustic perturbation family
- Source dataset and label confidence

## Data Construction Plan

1. **Seed from real datasets** — normalize local WearVox and HumDial-FDBench examples
2. **Normalize into multi-axis taxonomy** — map each item to behavior-setting class, participation, action, and acoustic labels
3. **Script scenario templates** — create controlled interaction scripts for each behavior setting
4. **Synthesize multi-speaker scenes** — target user, ratified participants, bystanders, overhearers, interrupters, joiners
5. **Script event-triggered monitoring templates** — user-set triggers with silence/response windows and distractor events
6. **Generate perturbation variants** — clean, mild, medium, severe using RIRs, noise mixing, overlap control, codec simulation
7. **Annotate each item** — expected action, addressed speaker, target answer, participation frame, trigger metadata, perturbation metadata
8. **Split by generalization** — separate speakers, environments, perturbation sources, behavior settings, scenario templates across train/dev/test partitions

## Relation to Existing Work

- **WearVox** — egocentric, multi-channel wearable assistant audio with side-talk rejection. Only `non-assistant-directed` rows are used; QA, tool-calling, and translation rows are excluded.
- **HumDial-FDBench** — real full-duplex test folders for participation and turn-taking labels. Folder names support Axis B; Axis A requires transcript-level audit.
- **Speak or Stay Silent** — context-aware turn-taking and speak-vs-silence decisions in multi-party dialogue.
- **SocialOmni** — who/when/how diagnostic layer for multi-party interaction failures.
- **Omni-DuplexEval** — event-triggered monitoring tasks where the assistant remains silent until a specified event.

Interaction-Bench differs by combining: audio-to-answer evaluation, addressee and participation detection, must-ignore vs must-respond vs must-wait behavior, event-triggered monitoring with explicit silence/response windows, who/when/how failure diagnosis, a near-exhaustive non-overlapping behavior-setting taxonomy, and controlled acoustic perturbation sweeps.

## Success Criteria

The first version succeeds if it can answer:

1. Which models incorrectly respond to side conversations, bystanders, or overheard public speech?
2. Which models fail to respond when a user or newly ratified participant addresses the assistant?
3. Which behavior settings are hardest across the 10-class taxonomy?
4. How much do reverb, overlap, noise, egocentric motion, and codec degradation reduce accuracy?
5. Are failures caused more by acoustic corruption, participation/addressee confusion, or their interaction?
6. Can models remain silent while monitoring for a user-defined event, then respond within the correct temporal window?
7. When models fail, is the failure primarily a who error, a when error, or a how error?

## First-Version Non-Goals

- No model training or fine-tuning
- No transcript benchmark as the primary objective — transcripts can appear as metadata, but scoring targets interaction behavior and answer quality
