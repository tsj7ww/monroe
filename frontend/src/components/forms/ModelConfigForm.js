import React, { useState, useCallback, useMemo } from 'react';
import { 
  Brain, Database, BarChart, ChevronDown, ChevronUp, AlertCircle 
} from 'lucide-react';

// Constants
const TASK_TYPES = {
  CLASSIFICATION: 'classification',
  REGRESSION: 'regression'
};

const METRICS = {
  CLASSIFICATION: [
    { value: 'accuracy', label: 'Accuracy' },
    { value: 'precision', label: 'Precision' },
    { value: 'recall', label: 'Recall' },
    { value: 'f1', label: 'F1 Score' },
    { value: 'roc_auc', label: 'ROC AUC' }
  ],
  REGRESSION: [
    { value: 'rmse', label: 'RMSE' },
    { value: 'mae', label: 'MAE' },
    { value: 'r2', label: 'R² Score' },
    { value: 'mape', label: 'MAPE' }
  ]
};

const SCALING_METHODS = [
  { value: 'none', label: 'No Scaling' },
  { value: 'standard', label: 'Standard Scaler' },
  { value: 'minmax', label: 'Min-Max Scaler' },
  { value: 'robust', label: 'Robust Scaler' }
];

const MISSING_VALUES = [
  { value: 'drop', label: 'Drop Rows' },
  { value: 'mean', label: 'Fill with Mean' },
  { value: 'median', label: 'Fill with Median' },
  { value: 'mode', label: 'Fill with Mode' },
  { value: 'ffill', label: 'Forward Fill' },
  { value: 'bfill', label: 'Backward Fill' }
];

const SEARCH_TYPES = [
  { value: 'grid', label: 'Grid Search' },
  { value: 'random', label: 'Random Search' },
  { value: 'bayesian', label: 'Bayesian Optimization' }
];

// Model definitions with hyperparameters
const modelDefinitions = {
  random_forest: {
    label: 'Random Forest',
    description: 'Ensemble of decision trees',
    hyperparameters: [
      { name: 'n_estimators', label: 'Number of Trees', type: 'range', default: '100', placeholder: '100 or 50,100,200' },
      { name: 'max_depth', label: 'Max Depth', type: 'range', default: '10', placeholder: '10 or 5,10,20' },
      { name: 'min_samples_split', label: 'Min Samples Split', type: 'range', default: '2', placeholder: '2 or 2,5,10' },
      { name: 'min_samples_leaf', label: 'Min Samples Leaf', type: 'range', default: '1', placeholder: '1 or 1,2,4' }
    ]
  },
  xgboost: {
    label: 'XGBoost',
    description: 'Gradient boosting framework',
    hyperparameters: [
      { name: 'n_estimators', label: 'Number of Rounds', type: 'range', default: '100', placeholder: '100 or 50,100,200' },
      { name: 'learning_rate', label: 'Learning Rate', type: 'range', default: '0.1', placeholder: '0.1 or 0.01,0.1,0.3' },
      { name: 'max_depth', label: 'Max Depth', type: 'range', default: '6', placeholder: '6 or 3,6,10' },
      { name: 'subsample', label: 'Subsample Ratio', type: 'range', default: '0.8', placeholder: '0.8 or 0.6,0.8,1.0' }
    ]
  },
  logistic_regression: {
    label: 'Logistic Regression',
    description: 'Linear model for classification',
    hyperparameters: [
      { name: 'C', label: 'Regularization (C)', type: 'range', default: '1.0', placeholder: '1.0 or 0.1,1.0,10.0' },
      { name: 'penalty', label: 'Penalty', type: 'select', options: ['l1', 'l2', 'elasticnet', 'none'], default: 'l2' },
      { name: 'solver', label: 'Solver', type: 'select', options: ['lbfgs', 'liblinear', 'saga'], default: 'lbfgs' },
      { name: 'max_iter', label: 'Max Iterations', type: 'range', default: '100', placeholder: '100 or 100,200,500' }
    ]
  },
  svm: {
    label: 'Support Vector Machine',
    description: 'Non-linear classification/regression',
    hyperparameters: [
      { name: 'C', label: 'Regularization (C)', type: 'range', default: '1.0', placeholder: '1.0 or 0.1,1.0,10.0' },
      { name: 'kernel', label: 'Kernel', type: 'select', options: ['linear', 'poly', 'rbf', 'sigmoid'], default: 'rbf' },
      { name: 'gamma', label: 'Gamma', type: 'select', options: ['scale', 'auto'], default: 'scale' },
      { name: 'degree', label: 'Polynomial Degree', type: 'range', default: '3', placeholder: '3 (only for poly kernel)' }
    ]
  },
  neural_network: {
    label: 'Neural Network',
    description: 'Deep learning model',
    hyperparameters: [
      { name: 'hidden_layers', label: 'Hidden Layers', type: 'text', default: '100,50', placeholder: '100,50 or (100,50),(200,100)' },
      { name: 'activation', label: 'Activation', type: 'select', options: ['relu', 'tanh', 'sigmoid'], default: 'relu' },
      { name: 'learning_rate', label: 'Learning Rate', type: 'range', default: '0.001', placeholder: '0.001 or 0.0001,0.001,0.01' },
      { name: 'batch_size', label: 'Batch Size', type: 'range', default: '32', placeholder: '32 or 16,32,64' },
      { name: 'epochs', label: 'Epochs', type: 'range', default: '100', placeholder: '100 or 50,100,200' }
    ]
  }
};

// Simple Input Component
const Input = ({ label, type = 'text', value, onChange, error, placeholder, min, max, step, options, required, tooltip }) => {
  const inputClasses = `w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
    error ? 'border-red-500' : 'border-gray-300'
  }`;

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
        {tooltip && <span className="text-xs text-gray-500 ml-2">({tooltip})</span>}
      </label>
      
      {type === 'select' ? (
        <select value={value} onChange={onChange} className={inputClasses}>
          {options?.map(opt => (
            <option key={opt.value || opt} value={opt.value || opt}>
              {opt.label || opt}
            </option>
          ))}
        </select>
      ) : type === 'checkbox' ? (
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={value}
            onChange={(e) => onChange({ target: { value: e.target.checked } })}
            className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
          />
          <span className="ml-2 text-sm text-gray-700">{placeholder}</span>
        </label>
      ) : (
        <input
          type={type}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          min={min}
          max={max}
          step={step}
          className={inputClasses}
        />
      )}
      
      {error && (
        <p className="mt-1 text-sm text-red-600 flex items-center">
          <AlertCircle className="w-4 h-4 mr-1" />
          {error}
        </p>
      )}
    </div>
  );
};

// Hyperparameter Input Component
const HyperparameterInput = ({ param, value, onChange }) => {
  if (param.type === 'select') {
    return (
      <Input
        label={param.label}
        type="select"
        value={value || param.default}
        onChange={(e) => onChange(e.target.value)}
        options={param.options}
      />
    );
  }
  
  return (
    <Input
      label={param.label}
      type="text"
      value={value || param.default}
      onChange={(e) => onChange(e.target.value)}
      placeholder={param.placeholder}
      tooltip={param.type === 'range' ? 'Single value or comma-separated list' : ''}
    />
  );
};

// Model Selection Card with Hyperparameters
const ModelCard = ({ modelKey, model, selected, hyperparameters, onToggle, onHyperparameterChange }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`border rounded-lg transition-colors ${selected ? 'border-blue-500 bg-blue-50' : 'border-gray-200'}`}>
      <div className="p-3">
        <label className="flex items-start cursor-pointer">
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggle}
            className="mt-1 w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
          />
          <div className="ml-3 flex-1">
            <div className="font-medium text-gray-900">{model.label}</div>
            <div className="text-sm text-gray-500">{model.description}</div>
            {selected && (
              <div className="mt-2">
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setExpanded(!expanded);
                  }}
                  className="inline-flex items-center px-3 py-1 text-sm font-medium text-blue-700 bg-blue-100 rounded-md hover:bg-blue-200 transition-colors"
                >
                  {expanded ? (
                    <>
                      <ChevronUp className="w-4 h-4 mr-1" />
                      Hide Hyperparameters
                    </>
                  ) : (
                    <>
                      <ChevronDown className="w-4 h-4 mr-1" />
                      Configure Hyperparameters
                    </>
                  )}
                </button>
                <p className="mt-1 text-xs text-gray-600">
                  {expanded 
                    ? "Specify single values or comma-separated lists for hyperparameter search" 
                    : "Click to customize model hyperparameters for better performance"
                  }
                </p>
              </div>
            )}
          </div>
        </label>
      </div>
      
      {selected && expanded && (
        <div className="px-3 pb-3 border-t border-gray-200 mt-3 pt-3 bg-gray-50">
          <div className="mb-2">
            <h4 className="text-sm font-semibold text-gray-700">Hyperparameter Configuration</h4>
            <p className="text-xs text-gray-600 mt-1">Enter single values or multiple values separated by commas for grid/random search</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {model.hyperparameters.map((param) => (
              <HyperparameterInput
                key={param.name}
                param={param}
                value={hyperparameters?.[param.name]}
                onChange={(value) => onHyperparameterChange(modelKey, param.name, value)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Main Component
const ModelConfigForm = ({ onConfigChange, initialConfig = {} }) => {
  // Default configuration
  const defaultConfig = {
    // Model config
    taskType: TASK_TYPES.CLASSIFICATION,
    selectedModels: {},
    modelHyperparameters: {},
    useHyperparameterTuning: true,
    searchType: 'grid',
    searchIterations: 10,
    
    // Data config
    targetColumn: '',
    featureColumns: '',
    testSize: 0.2,
    validationSize: 0.1,
    randomState: 42,
    scalingMethod: 'standard',
    handleMissingValues: 'drop',
    categoricalEncoding: 'onehot',
    featureSelection: false,
    featureSelectionN: 10,
    
    // Evaluation config
    primaryMetric: 'accuracy',
    additionalMetrics: [],
    crossValidation: 5,
    stratifiedSplit: true,
    earlyStoppingRounds: 10,
    generateReport: true,
    saveModels: true,
    
    ...initialConfig
  };

  const [config, setConfig] = useState(defaultConfig);
  const [errors, setErrors] = useState({});

  // Update config and notify parent
  const updateConfig = useCallback((updates) => {
    const newConfig = { ...config, ...updates };
    setConfig(newConfig);
    
    // Validate
    const newErrors = {};
    if (Object.keys(newConfig.selectedModels).filter(k => newConfig.selectedModels[k]).length === 0) {
      newErrors.models = 'Select at least one model';
    }
    if (!newConfig.targetColumn) {
      newErrors.targetColumn = 'Required';
    }
    if (newConfig.testSize + newConfig.validationSize >= 0.9) {
      newErrors.splits = 'Test + validation size must be less than 90%';
    }
    setErrors(newErrors);
    
    // Notify parent
    if (onConfigChange) {
      onConfigChange(newConfig, Object.keys(newErrors).length === 0);
    }
  }, [config, onConfigChange]);

  // Handle task type change
  const handleTaskTypeChange = useCallback((taskType) => {
    const metrics = taskType === TASK_TYPES.CLASSIFICATION 
      ? METRICS.CLASSIFICATION 
      : METRICS.REGRESSION;
    
    updateConfig({
      taskType,
      primaryMetric: metrics[0].value,
      additionalMetrics: []
    });
  }, [updateConfig]);

  // Handle hyperparameter change
  const handleHyperparameterChange = useCallback((modelKey, paramName, value) => {
    updateConfig({
      modelHyperparameters: {
        ...config.modelHyperparameters,
        [modelKey]: {
          ...config.modelHyperparameters[modelKey],
          [paramName]: value
        }
      }
    });
  }, [config.modelHyperparameters, updateConfig]);

  // Get available metrics based on task type
  const availableMetrics = useMemo(() => 
    config.taskType === TASK_TYPES.CLASSIFICATION 
      ? METRICS.CLASSIFICATION 
      : METRICS.REGRESSION,
    [config.taskType]
  );

  return (
    <div className="space-y-6">
      {/* Model Configuration */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center mb-4">
          <Brain className="w-5 h-5 text-purple-600 mr-2" />
          <h3 className="text-lg font-semibold">Model Configuration</h3>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Task Type"
              type="select"
              value={config.taskType}
              onChange={(e) => handleTaskTypeChange(e.target.value)}
              options={[
                { value: TASK_TYPES.CLASSIFICATION, label: 'Classification' },
                { value: TASK_TYPES.REGRESSION, label: 'Regression' }
              ]}
              required
            />
            
            <Input
              label="Hyperparameter Search"
              type="select"
              value={config.searchType}
              onChange={(e) => updateConfig({ searchType: e.target.value })}
              options={SEARCH_TYPES}
            />
          </div>

          {config.searchType !== 'grid' && (
            <Input
              label="Search Iterations"
              type="number"
              value={config.searchIterations}
              onChange={(e) => updateConfig({ searchIterations: parseInt(e.target.value) })}
              min="5"
              max="100"
              placeholder="Number of parameter combinations to try"
            />
          )}
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Models <span className="text-red-500">*</span>
            </label>
            {errors.models && (
              <p className="mb-2 text-sm text-red-600 flex items-center">
                <AlertCircle className="w-4 h-4 mr-1" />
                {errors.models}
              </p>
            )}
            <div className="space-y-2">
              {Object.entries(modelDefinitions).map(([key, model]) => (
                <ModelCard
                  key={key}
                  modelKey={key}
                  model={model}
                  selected={config.selectedModels[key] || false}
                  hyperparameters={config.modelHyperparameters[key]}
                  onToggle={() => updateConfig({
                    selectedModels: {
                      ...config.selectedModels,
                      [key]: !config.selectedModels[key]
                    }
                  })}
                  onHyperparameterChange={handleHyperparameterChange}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Data Configuration */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center mb-4">
          <Database className="w-5 h-5 text-green-600 mr-2" />
          <h3 className="text-lg font-semibold">Data Configuration</h3>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Target Column"
              value={config.targetColumn}
              onChange={(e) => updateConfig({ targetColumn: e.target.value })}
              placeholder="e.g., target, label, y"
              error={errors.targetColumn}
              required
            />
            
            <Input
              label="Feature Columns"
              value={config.featureColumns}
              onChange={(e) => updateConfig({ featureColumns: e.target.value })}
              placeholder="Leave empty for all columns or specify: col1,col2,col3"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Input
              label="Test Size"
              type="number"
              value={config.testSize}
              onChange={(e) => updateConfig({ testSize: parseFloat(e.target.value) })}
              min="0.1"
              max="0.5"
              step="0.05"
              error={errors.splits}
            />
            
            <Input
              label="Validation Size"
              type="number"
              value={config.validationSize}
              onChange={(e) => updateConfig({ validationSize: parseFloat(e.target.value) })}
              min="0"
              max="0.3"
              step="0.05"
              tooltip="Set to 0 for no validation set"
            />
            
            <Input
              label="Random State"
              type="number"
              value={config.randomState}
              onChange={(e) => updateConfig({ randomState: parseInt(e.target.value) })}
              placeholder="42"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Feature Scaling"
              type="select"
              value={config.scalingMethod}
              onChange={(e) => updateConfig({ scalingMethod: e.target.value })}
              options={SCALING_METHODS}
            />
            
            <Input
              label="Handle Missing Values"
              type="select"
              value={config.handleMissingValues}
              onChange={(e) => updateConfig({ handleMissingValues: e.target.value })}
              options={MISSING_VALUES}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Categorical Encoding"
              type="select"
              value={config.categoricalEncoding}
              onChange={(e) => updateConfig({ categoricalEncoding: e.target.value })}
              options={[
                { value: 'onehot', label: 'One-Hot Encoding' },
                { value: 'label', label: 'Label Encoding' },
                { value: 'target', label: 'Target Encoding' },
                { value: 'ordinal', label: 'Ordinal Encoding' }
              ]}
            />
            
            <div className="space-y-2">
              <Input
                label="Feature Selection"
                type="checkbox"
                value={config.featureSelection}
                onChange={(e) => updateConfig({ featureSelection: e.target.value })}
                placeholder="Enable automatic feature selection"
              />
              {config.featureSelection && (
                <Input
                  label="Number of Features"
                  type="number"
                  value={config.featureSelectionN}
                  onChange={(e) => updateConfig({ featureSelectionN: parseInt(e.target.value) })}
                  min="5"
                  max="100"
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Evaluation Configuration */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center mb-4">
          <BarChart className="w-5 h-5 text-blue-600 mr-2" />
          <h3 className="text-lg font-semibold">Evaluation Configuration</h3>
        </div>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Primary Metric"
              type="select"
              value={config.primaryMetric}
              onChange={(e) => updateConfig({ primaryMetric: e.target.value })}
              options={availableMetrics}
            />
            
            <Input
              label="Cross Validation Folds"
              type="number"
              value={config.crossValidation}
              onChange={(e) => updateConfig({ crossValidation: parseInt(e.target.value) })}
              min="2"
              max="10"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Additional Metrics to Track
            </label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {availableMetrics
                .filter(m => m.value !== config.primaryMetric)
                .map(metric => (
                  <label key={metric.value} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={config.additionalMetrics.includes(metric.value)}
                      onChange={(e) => {
                        const metrics = e.target.checked
                          ? [...config.additionalMetrics, metric.value]
                          : config.additionalMetrics.filter(m => m !== metric.value);
                        updateConfig({ additionalMetrics: metrics });
                      }}
                      className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-700">{metric.label}</span>
                  </label>
                ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Early Stopping Rounds"
              type="number"
              value={config.earlyStoppingRounds}
              onChange={(e) => updateConfig({ earlyStoppingRounds: parseInt(e.target.value) })}
              min="0"
              max="50"
              tooltip="0 to disable early stopping"
            />
            
            {config.taskType === TASK_TYPES.CLASSIFICATION && (
              <Input
                label="Stratified Split"
                type="checkbox"
                value={config.stratifiedSplit}
                onChange={(e) => updateConfig({ stratifiedSplit: e.target.value })}
                placeholder="Maintain class distribution in splits"
              />
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Generate Report"
              type="checkbox"
              value={config.generateReport}
              onChange={(e) => updateConfig({ generateReport: e.target.value })}
              placeholder="Generate detailed evaluation report"
            />
            
            <Input
              label="Save Models"
              type="checkbox"
              value={config.saveModels}
              onChange={(e) => updateConfig({ saveModels: e.target.value })}
              placeholder="Save trained models to disk"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelConfigForm;