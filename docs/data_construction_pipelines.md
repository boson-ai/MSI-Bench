# Data Construction Pipelines

Interaction-Bench can use two complementary construction routes: a controlled
synthetic scenario pipeline and a naturalistic real-recording pipeline.

## Route A: Synthetic Scenario Pipeline

```mermaid
graph TD
    A[Planner / Scenario Sampler] --> B[Scenario Spec]
    B --> C[Dialogue and Event Script]
    C --> D[Metadata and Rubric Freeze]
    D --> E[TTS / Audio Asset Generation]
    E --> F[Mixing and Acoustic Perturbation]
    F --> G[Dataset QC]
    G --> H{QC Pass?}
    H -- No: script issue --> C
    H -- No: label issue --> D
    H -- No: audio issue --> E
    H -- No: mixing issue --> F
    H -- Yes --> I[Benchmark Manifest]
    I --> J[Model Prediction]
    J --> K[Scoring / Judge]

    A -. samples .-> A1[Axis A: Behavior Setting]
    A -. samples .-> A2[Axis B: Participation Framework]
    A -. samples .-> A3[Axis C: Expected Assistant Action]
    A -. optional .-> A4[Axis D: Trigger and Temporal Context]
    F -. applies .-> A5[Axis E: Acoustic Robustness]

    D -. freezes .-> R[Success Rubric]
    D -. freezes .-> M[Expected Action, Speaker Roles, Timing Windows]
    C -. defines .-> T[Target Events and Distractor Events]
    F -. outputs .-> U[Clean Reference, Mixed Audio, Timing Map]
    G -. verifies .-> V[Audio Quality, Trigger Timing, Label Clarity]
    K -. reports .-> W[Engagement, Trigger, Robustness, Who/When/How Metrics]
```

## Route B: Real Recording Pipeline

```mermaid
graph TD
    A[Raw Real Recordings] --> B[Sommelier-style Preprocessing]
    B --> C[Structured Conversation Record]
    C --> D[Metadata Annotation]
    D --> E[Dataset QC / Human Audit]
    E --> F{QC Pass?}
    F -- No: preprocessing issue --> B
    F -- No: ambiguous label --> D
    F -- No: unsuitable sample --> X[Exclude or Hold for Audit]
    F -- Yes --> G[Benchmark Manifest]
    G --> H[Model Prediction]
    H --> I[Scoring / Judge]

    B -. includes .-> B1[VAD]
    B -. includes .-> B2[Speaker Diarization]
    B -. includes .-> B3[Overlap Separation]
    B -. includes .-> B4[ASR Ensemble]
    B -. includes .-> B5[Timestamp Alignment]
    B -. includes .-> B6[Noise / BGM Handling]

    C -. contains .-> C1[Speaker Turns]
    C -. contains .-> C2[Overlap Regions]
    C -. contains .-> C3[Transcript Candidates]
    C -. contains .-> C4[Audio Quality Flags]

    D -. adds .-> D1[Axis A: Behavior Setting]
    D -. adds .-> D2[Axis B: Participation Framework]
    D -. adds .-> D3[Axis C: Expected Assistant Action]
    D -. optional .-> D4[Axis D: Trigger and Temporal Context]
    D -. adds .-> D5[Axis E: Acoustic Robustness]
    D -. adds .-> D6[Label Confidence and Rationale]

    E -. verifies .-> E1[Speaker Attribution]
    E -. verifies .-> E2[Transcript Sanity]
    E -. verifies .-> E3[Action Unambiguity]
    E -. verifies .-> E4[Privacy and License]
    I -. reports .-> I1[Engagement, False Engagement, Missed Engagement, Who/When/How Metrics]
```
