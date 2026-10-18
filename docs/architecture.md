# Architecture

## Hybrid Edge-Enabled Client–Server Modular Monolith Architecture

The system keeps latency-sensitive and privacy-sensitive perception and hardware
control on the computer attached to the camera and robot. A FastAPI modular
monolith exposes immediate REST operations, coordinates module use cases, emits
internal events or background jobs, and initially persists through one shared
database. Module boundaries are explicit so an infrastructure adapter can later
move without changing domain rules.

```mermaid
flowchart LR
    Mobile[Android child app\nMVVM] -->|start/session REST| API[FastAPI API layer]
    Dashboard[Teacher dashboard\ncomponents] -->|profiles/reports REST| API

    subgraph Edge[Hardware-connected edge computer]
        Camera[Camera] --> Cube[Cube detection +\nanswer validation]
        Camera --> Behaviour[Hand, head pose +\ncoarse gaze analysis]
        Robot[Robotic arm] --- RobotControl[Safe robot control]
        Cube --> RobotControl
    end

    API --> Assessment[Assessment]
    Assessment --> Adaptive[Adaptive questions]
    Adaptive -->|task + difficulty| Mobile
    Mobile -->|task identifiers| Edge
    Cube -->|derived cube result REST| API
    Behaviour -->|derived observation REST| API

    API --> Orchestrator[Assessment orchestrator]
    Orchestrator --> Risk[Risk profiling]
    Risk -->|internal event / background task| Profiles[Student profiles]
    Profiles --> Reporting[Reporting]

    Assessment --> DB[(Shared database)]
    Adaptive --> DB
    Risk --> DB
    Profiles --> DB
    Reporting --> DB

    Training[Offline training workspaces] -. versioned model artifacts .-> Edge
    Training -. versioned model artifacts .-> Adaptive
    Training -. versioned model artifacts .-> Risk
```

## Module layering

Every research module follows this dependency direction:

```text
API -> Service -> Domain Logic -> Infrastructure ports/adapters
```

- `api.py` declares FastAPI routing only.
- `schemas.py` exposes request/response contracts.
- `service.py` implements application use cases and dependency injection.
- `domain/` owns framework-independent rules.
- specialised folders implement computer vision, model loading, persistence, or
  hardware adapters.
- `models/` loads versioned artifacts; it does not train them.

Cross-module coordination belongs in `backend/app/orchestration/`, not inside a
research module. Cross-module payloads belong in `shared/contracts/`.

## Deployment boundary

Camera capture, MediaPipe, cube detection, behaviour classification, and robotic
control must run on the edge computer. Continuous raw video upload is not
required. The backend receives derived, pseudonymous observations whenever
possible. If the edge and server run on one machine during development, this
logical boundary still applies.

The initial shared database simplifies transactions and operations. Modules must
access data through owned repository interfaces so shared storage does not become
shared business logic.
