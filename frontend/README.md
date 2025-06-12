# React

Assets - Static files like images, icons, fonts, CSS files, and any pre-trained model files or datasets. These don't contain logic but are resources your app needs.

Components - Reusable UI building blocks that encapsulate specific functionality. In an ML app, you might have components like ModelVisualizer, DataUploader, PredictionDisplay, or ParameterSlider. These are the Lego blocks of your interface.

Pages - Top-level route components that represent entire screens or views. Examples might include Dashboard, ModelTraining, DataExploration, or Results. These compose multiple components together.

Context - React Context providers that manage global state across your app. For ML applications, you might have contexts for ModelState (current model parameters), DataContext (loaded datasets), or UserPreferences. This avoids prop drilling for shared data.

Hooks - Custom React hooks that encapsulate reusable stateful logic. In ML apps, you might create hooks like useModelPrediction, useDataProcessing, useModelTraining, or useVisualization. These let you share complex logic between components.

Services - Functions that handle external operations like API calls, data processing, or model inference. Examples include modelAPI.js for backend communication, dataProcessor.js for cleaning datasets, or tensorflowService.js for running models. These abstract away implementation details.

Utils - Pure utility functions that perform common operations without side effects. Think formatNumbers, validateInput, calculateMetrics, or parseCSV. These are helper functions used throughout your app.