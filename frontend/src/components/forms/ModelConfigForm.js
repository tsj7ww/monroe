import React, { useState, useCallback } from 'react';
import { Download, RotateCcw, Settings, Database, Brain, Sliders, CheckCircle, AlertCircle } from 'lucide-react';

const InputField = React.memo(({ 
  label, 
  name, 
  type = 'text', 
  min, 
  max, 
  step, 
  options, 
  placeholder, 
  required = false, 
  icon: Icon, 
  value, 
  onChange, 
  error 
}) => (
  <div className="space-y-2">
    <label className="flex items-center text-sm font-medium text-gray-700">
      {Icon && <Icon className="w-4 h-4 mr-2 text-gray-500" />}
      {label}
      {required && <span className="text-red-500 ml-1">*</span>}
    </label>
    
    {type === 'select' ? (
      <select
        name={name}
        value={value}
        onChange={onChange}
        className={`w-full px-3 py-2 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
          error ? 'border-red-500' : 'border-gray-300'
        }`}
      >
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    ) : type === 'checkbox' ? (
      <label className="flex items-center space-x-3 cursor-pointer">
        <input
          type="checkbox"
          name={name}
          checked={value}
          onChange={onChange}
          className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
        />
        <span className="text-sm text-gray-700">{label}</span>
      </label>
    ) : (
      <input
        type={type}
        name={name}
        value={value}
        onChange={onChange}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        className={`w-full px-3 py-2 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors ${
          error ? 'border-red-500' : 'border-gray-300'
        }`}
      />
    )}
    
    {error && (
      <p className="flex items-center text-sm text-red-600">
        <AlertCircle className="w-4 h-4 mr-1" />
        {error}
      </p>
    )}
  </div>
));

const Section = React.memo(({ title, icon: Icon, children }) => (
  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
    <div className="flex items-center mb-4">
      <Icon className="w-5 h-5 text-blue-600 mr-2" />
      <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
    </div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {children}
    </div>
  </div>
));

const initialConfig = {
  modelName: '',
  modelType: 'random_forest',
  taskType: 'classification',
  targetColumn: '',
  testSize: 0.2,
  randomState: 42,
  maxDepth: 10,
  nEstimators: 100,
  learningRate: 0.1,
  regularization: 0.01,
  crossValidation: 5,
  scoringMetric: 'accuracy',
  earlyStoppingRounds: 10,
  featureSelection: false,
  hyperparameterTuning: false,
  classWeight: 'balanced'
};

const modelTypeOptions = [
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'xgboost', label: 'XGBoost' },
  { value: 'neural_network', label: 'Neural Network' },
  { value: 'logistic_regression', label: 'Logistic Regression' },
  { value: 'svm', label: 'Support Vector Machine' }
];

const taskTypeOptions = [
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' }
];

const classWeightOptions = [
  { value: 'balanced', label: 'Balanced' },
  { value: 'none', label: 'None' }
];

const scoringMetricOptions = [
  { value: 'accuracy', label: 'Accuracy' },
  { value: 'precision', label: 'Precision' },
  { value: 'recall', label: 'Recall' },
  { value: 'f1', label: 'F1 Score' },
  { value: 'roc_auc', label: 'ROC AUC' }
];

const ModelConfigForm = () => {
  const [config, setConfig] = useState(initialConfig);
  const [jsonOutput, setJsonOutput] = useState('');
  const [showJson, setShowJson] = useState(false);
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validateForm = useCallback(() => {
    const newErrors = {};
    
    if (!config.modelName.trim()) {
      newErrors.modelName = 'Model name is required';
    }
    
    if (!config.targetColumn.trim()) {
      newErrors.targetColumn = 'Target column is required';
    }
    
    if (config.testSize < 0.1 || config.testSize > 0.5) {
      newErrors.testSize = 'Test size must be between 0.1 and 0.5';
    }
    
    if (config.maxDepth < 1 || config.maxDepth > 50) {
      newErrors.maxDepth = 'Max depth must be between 1 and 50';
    }
    
    if (config.nEstimators < 10 || config.nEstimators > 1000) {
      newErrors.nEstimators = 'N estimators must be between 10 and 1000';
    }
    
    if (config.learningRate < 0.001 || config.learningRate > 1) {
      newErrors.learningRate = 'Learning rate must be between 0.001 and 1';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [config]);

  const handleInputChange = useCallback((e) => {
    const { name, value, type, checked } = e.target;
    
    setConfig(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : 
               type === 'number' ? (value === '' ? '' : parseFloat(value)) : value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  }, [errors]);

  const generateJson = useCallback(() => {
    const jsonString = JSON.stringify(config, null, 2);
    setJsonOutput(jsonString);
    setShowJson(true);
  }, [config]);

  const handleSubmit = useCallback(async () => {
    if (!validateForm()) {
      return;
    }
    
    setIsSubmitting(true);
    
    // Simulate API call
    setTimeout(() => {
      console.log('Configuration generated:', config);
      setIsSubmitting(false);
      generateJson();
    }, 1000);
  }, [validateForm, config, generateJson]);

  const downloadJson = useCallback(() => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(jsonOutput);
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `${config.modelName || 'model_config'}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  }, [jsonOutput, config.modelName]);

  const resetForm = useCallback(() => {
    setConfig(initialConfig);
    setShowJson(false);
    setJsonOutput('');
    setErrors({});
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Model Configuration Generator
          </h1>
          <p className="text-gray-600">
            Configure your machine learning model parameters with ease
          </p>
        </div>
        
        <div className="space-y-6">
          {/* Basic Settings */}
          <Section title="Basic Settings" icon={Settings}>
            <InputField
              label="Model Name"
              name="modelName"
              placeholder="e.g., fraud_detection_v1"
              required
              icon={Brain}
              value={config.modelName}
              onChange={handleInputChange}
              error={errors.modelName}
            />
            
            <InputField
              label="Model Type"
              name="modelType"
              type="select"
              value={config.modelType}
              onChange={handleInputChange}
              error={errors.modelType}
              options={modelTypeOptions}
            />
            
            <InputField
              label="Task Type"
              name="taskType"
              type="select"
              value={config.taskType}
              onChange={handleInputChange}
              error={errors.taskType}
              options={taskTypeOptions}
            />
            
            <InputField
              label="Target Column"
              name="targetColumn"
              placeholder="e.g., is_fraud"
              required
              icon={Database}
              value={config.targetColumn}
              onChange={handleInputChange}
              error={errors.targetColumn}
            />
          </Section>

          {/* Data Settings */}
          <Section title="Data Settings" icon={Database}>
            <InputField
              label="Test Size"
              name="testSize"
              type="number"
              min="0.1"
              max="0.5"
              step="0.05"
              value={config.testSize}
              onChange={handleInputChange}
              error={errors.testSize}
            />
            
            <InputField
              label="Random State"
              name="randomState"
              type="number"
              value={config.randomState}
              onChange={handleInputChange}
              error={errors.randomState}
            />
            
            <InputField
              label="Class Weight"
              name="classWeight"
              type="select"
              value={config.classWeight}
              onChange={handleInputChange}
              error={errors.classWeight}
              options={classWeightOptions}
            />
          </Section>

          {/* Model Parameters */}
          <Section title="Model Parameters" icon={Sliders}>
            <InputField
              label="Max Depth"
              name="maxDepth"
              type="number"
              min="1"
              max="50"
              value={config.maxDepth}
              onChange={handleInputChange}
              error={errors.maxDepth}
            />
            
            <InputField
              label="N Estimators"
              name="nEstimators"
              type="number"
              min="10"
              max="1000"
              step="10"
              value={config.nEstimators}
              onChange={handleInputChange}
              error={errors.nEstimators}
            />
            
            <InputField
              label="Learning Rate"
              name="learningRate"
              type="number"
              min="0.001"
              max="1"
              step="0.01"
              value={config.learningRate}
              onChange={handleInputChange}
              error={errors.learningRate}
            />
            
            <InputField
              label="Regularization"
              name="regularization"
              type="number"
              min="0"
              max="1"
              step="0.001"
              value={config.regularization}
              onChange={handleInputChange}
              error={errors.regularization}
            />
          </Section>

          {/* Training Settings */}
          <Section title="Training Settings" icon={Brain}>
            <InputField
              label="Cross Validation Folds"
              name="crossValidation"
              type="number"
              min="2"
              max="10"
              value={config.crossValidation}
              onChange={handleInputChange}
              error={errors.crossValidation}
            />
            
            <InputField
              label="Scoring Metric"
              name="scoringMetric"
              type="select"
              value={config.scoringMetric}
              onChange={handleInputChange}
              error={errors.scoringMetric}
              options={scoringMetricOptions}
            />
            
            <InputField
              label="Early Stopping Rounds"
              name="earlyStoppingRounds"
              type="number"
              min="5"
              max="50"
              value={config.earlyStoppingRounds}
              onChange={handleInputChange}
              error={errors.earlyStoppingRounds}
            />
          </Section>

          {/* Advanced Options */}
          <Section title="Advanced Options" icon={Settings}>
            <InputField
              label="Enable Feature Selection"
              name="featureSelection"
              type="checkbox"
              value={config.featureSelection}
              onChange={handleInputChange}
              error={errors.featureSelection}
            />
            
            <InputField
              label="Enable Hyperparameter Tuning"
              name="hyperparameterTuning"
              type="checkbox"
              value={config.hyperparameterTuning}
              onChange={handleInputChange}
              error={errors.hyperparameterTuning}
            />
          </Section>

          {/* Action Buttons */}
          <div className="flex flex-wrap justify-center gap-4 pt-6">
            <button
              type="button"
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg shadow-md hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Generating...
                </>
              ) : (
                <>
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Generate JSON
                </>
              )}
            </button>
            
            <button
              type="button"
              onClick={resetForm}
              className="flex items-center px-6 py-3 bg-gray-600 text-white font-medium rounded-lg shadow-md hover:bg-gray-700 focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset Form
            </button>
            
            {showJson && (
              <button
                type="button"
                onClick={downloadJson}
                className="flex items-center px-6 py-3 bg-green-600 text-white font-medium rounded-lg shadow-md hover:bg-green-700 focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-colors"
              >
                <Download className="w-4 h-4 mr-2" />
                Download JSON
              </button>
            )}
          </div>
        </div>

        {/* JSON Output */}
        {showJson && (
          <div className="mt-8 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
              Generated Configuration
            </h3>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <pre className="text-sm text-gray-800 overflow-auto max-h-96 whitespace-pre-wrap font-mono">
                {jsonOutput}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelConfigForm;