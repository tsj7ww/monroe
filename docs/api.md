```mermaid
flowchart TB
    api[API]
    data[
    Data
    - Process
    - Upload
    ]
    train[
    Train
    - Kickoff
    - Status
    - Results
    ]
    serve[
    Serve
    - Kickoff
    - Status
    - Results
    ]
    auth[
    Auth
    ]
    infra[Infra]
    api --> data
    api --> train
    api --> serve
    train --> infra
    serve --> infra
    api --> auth
```