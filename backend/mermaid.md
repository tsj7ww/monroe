```mermaid
flowchart TB
    data[
    Data
    ]
    preprocess[
    Preprocess
    ]
    selection[
    Selection
    ]
    train[
    Train
    - GLM
    - GBM
    - Neural Network
    ]
    evaluate[
    Evaluate
    - eval metric 1
    - eval metric 2
    ]
    monitor[
    Monitor
    ]
    enduser[End User]

	data -->|data| preprocess
	preprocess -->|data| selection
	selection -->|data| train
	train -->|data| evaluate
	evaluate -->|data| selection
	selection -->|data| monitor
	monitor -->|data| enduser
```