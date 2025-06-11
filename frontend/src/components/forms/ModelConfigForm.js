import React, { useState } from 'react';

const ModelConfigForm = () => {
  const [config, setConfig] = useState({
    // Basic Settings
    modelName: '',
    modelType: 'random_forest',
    taskType: 'classification',
    
    // Data Settings
    targetColumn: '',
    testSize: 0.2,
    randomState: 42,
    
    // Model Parameters
    maxDepth: 10,
    nEstimators: 100,
    learningRate: 0.1,
    regularization: 0.01,
    
    // Training Settings
    crossValidation: 5,
    scoringMetric: 'accuracy',
    earlyStoppingRounds: 10,
    
    // Advanced Options
    featureSelection: false,
    hyperparameterTuning: false,
    classWeight: 'balanced'
  });

  const [jsonOutput, setJsonOutput] = useState('');
  const [showJson, setShowJson] = useState(false);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : 
               type === 'number' ? parseFloat(value) : value
    }));
  };

  const generateJson = () => {
    const jsonString = JSON.stringify(config, null, 2);
    setJsonOutput(jsonString);
    setShowJson(true);
  };

  const downloadJson = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(jsonOutput);
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `${config.modelName || 'model_config'}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const resetForm = () => {
    setConfig({
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
    });
    setShowJson(false);
    setJsonOutput('');
  };

  const containerStyle = {
    padding: '20px',
    backgroundColor: '#f5f5f5',
    minHeight: '100vh',
    fontFamily: 'Arial, sans-serif'
  };

  const formStyle = {
    backgroundColor: 'white',
    padding: '30px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    maxWidth: '800px',
    margin: '0 auto'
  };

  const sectionStyle = {
    marginBottom: '30px',
    paddingBottom: '20px',
    borderBottom: '1px solid #eee'
  };

  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '20px',
    marginTop: '15px'
  };

  const inputGroupStyle = {
    marginBottom: '15px'
  };

  const labelStyle = {
    display: 'block',
    marginBottom: '5px',
    fontWeight: 'bold',
    color: '#333'
  };

  const inputStyle = {
    width: '100%',
    padding: '8px 12px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '14px'
  };

  const selectStyle = {
    ...inputStyle,
    backgroundColor: 'white'
  };

  const buttonStyle = {
    padding: '10px 20px',
    margin: '5px',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold'
  };

  const primaryButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#007bff',
    color: 'white'
  };

  const secondaryButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#6c757d',
    color: 'white'
  };

  const successButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#28a745',
    color: 'white'
  };

  const jsonStyle = {
    backgroundColor: '#f8f9fa',
    border: '1px solid #dee2e6',
    borderRadius: '4px',
    padding: '15px',
    fontFamily: 'monospace',
    fontSize: '12px',
    overflow: 'auto',
    maxHeight: '400px',
    marginTop: '20px'
  };

  return (
    <div style={containerStyle}>
      <h1 style={{ textAlign: 'center', marginBottom: '30px', color: '#333' }}>
        Model Configuration Generator
      </h1>
      
      <div style={formStyle}>
        {/* Basic Settings */}
        <div style={sectionStyle}>
          <h3 style={{ color: '#007bff', marginBottom: '15px' }}>Basic Settings</h3>
          <div style={gridStyle}>
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Model Name</label>
              <input
                type="text"
                name="modelName"
                value={config.modelName}
                onChange={handleInputChange}
                placeholder="e.g., fraud_detection_v1"
                style={inputStyle}
              />
            </div>
            
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Model Type</label>
              <select
                name="modelType"
                value={config.modelType}
                onChange={handleInputChange}
                style={selectStyle}
              >
                <option value="random_forest">Random Forest</option>
                <option value="xgboost">XGBoost</option>
                <option value="neural_network">Neural Network</option>
                <option value="logistic_regression">Logistic Regression</option>
                <option value="svm">Support Vector Machine</option>
              </select>
            </div>
            
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Task Type</label>
              <select
                name="taskType"
                value={config.taskType}
                onChange={handleInputChange}
                style={selectStyle}
              >
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>
            
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Target Column</label>
              <input
                type="text"
                name="targetColumn"
                value={config.targetColumn}
                onChange={handleInputChange}
                placeholder="e.g., is_fraud"
                style={inputStyle}
              />
            </div>
          </div>
        </div>

        {/* Data Settings */}
        <div style={sectionStyle}>
          <h3 style={{ color: '#007bff', marginBottom: '15px' }}>Data Settings</h3>
          <div style={gridStyle}>
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Test Size (0.1 - 0.5)</label>
              <input
                type="number"
                name="testSize"
                value={config.testSize}
                onChange={handleInputChange}
                min="0.1"
                max="0.5"
                step="0.05"
                style={inputStyle}
              />
            </div>
            
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Random State</label>
              <input
                type="number"
                name="randomState"
                value={config.randomState}
                onChange={handleInputChange}
                style={inputStyle}
              />
            </div>
            
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Class Weight</label>
              <select
                name="classWeight"
                value={config.classWeight}
                onChange={handleInputChange}
                style={selectStyle}
              >
                <option value="balanced">Balanced</option>
                <option value="none">None</option>
              </select>
            </div>
          </div>
        </div>

        {/* Model Parameters */}
        <div style={sectionStyle}>
          <h3 style={{ color: '#007bff', marginBottom: '15px' }}>Model Parameters</h3>
          <div style={gridStyle}>
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Max Depth</label>
              <input
                type="number"
                name="maxDepth"
                value={config.maxDepth}
                onChange={handleInputChange}
                min="1"
                max="50"
                style={inputStyle}
              />
            </div>
            
            <div style={inputGroupStyle}>
              <label style={labelStyle}>N Estimators</label>
              <input
                type="number"
                name="nEstimators"
                value={config.nEstimators}
                onChange={handleInputChange}
                min="10"
                max="1000"
                step="10"
                style={inputStyle}
              />
            </div>
            
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Learning Rate</label>
              <input
                type="number"
                name="learningRate"
                value={config.learningRate}
                onChange={handleInputChange}
                min="0.001"
                max="1"
                step="0.01"
                style={inputStyle}
              />
            </div>
            
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Regularization</label>
              <input
                type="number"
                name="regularization"
                value={config.regularization}
                onChange={handleInputChange}
                min="0"
                max="1"
                step="0.001"
                style={inputStyle}
              />
            </div>
          </div>
        </div>

        {/* Training Settings */}
        <div style={sectionStyle}>
          <h3 style={{ color: '#007bff', marginBottom: '15px' }}>Training Settings</h3>
          <div style={gridStyle}>
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Cross Validation Folds</label>
              <input
                type="number"
                name="crossValidation"
                value={config.crossValidation}
                onChange={handleInputChange}
                min="2"
                max="10"
                style={inputStyle}
              />
            </div>
            
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Scoring Metric</label>
              <select
                name="scoringMetric"
                value={config.scoringMetric}
                onChange={handleInputChange}
                style={selectStyle}
              >
                <option value="accuracy">Accuracy</option>
                <option value="precision">Precision</option>
                <option value="recall">Recall</option>
                <option value="f1">F1 Score</option>
                <option value="roc_auc">ROC AUC</option>
              </select>
            </div>
            
            <div style={inputGroupStyle}>
              <label style={labelStyle}>Early Stopping Rounds</label>
              <input
                type="number"
                name="earlyStoppingRounds"
                value={config.earlyStoppingRounds}
                onChange={handleInputChange}
                min="5"
                max="50"
                style={inputStyle}
              />
            </div>
          </div>
        </div>

        {/* Advanced Options */}
        <div style={sectionStyle}>
          <h3 style={{ color: '#007bff', marginBottom: '15px' }}>Advanced Options</h3>
          <div style={gridStyle}>
            <div style={inputGroupStyle}>
              <label style={{ ...labelStyle, display: 'flex', alignItems: 'center' }}>
                <input
                  type="checkbox"
                  name="featureSelection"
                  checked={config.featureSelection}
                  onChange={handleInputChange}
                  style={{ marginRight: '8px' }}
                />
                Enable Feature Selection
              </label>
            </div>
            
            <div style={inputGroupStyle}>
              <label style={{ ...labelStyle, display: 'flex', alignItems: 'center' }}>
                <input
                  type="checkbox"
                  name="hyperparameterTuning"
                  checked={config.hyperparameterTuning}
                  onChange={handleInputChange}
                  style={{ marginRight: '8px' }}
                />
                Enable Hyperparameter Tuning
              </label>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{ textAlign: 'center', marginTop: '30px' }}>
          <button
            onClick={generateJson}
            style={primaryButtonStyle}
          >
            Generate JSON
          </button>
          
          <button
            onClick={resetForm}
            style={secondaryButtonStyle}
          >
            Reset Form
          </button>
          
          {showJson && (
            <button
              onClick={downloadJson}
              style={successButtonStyle}
            >
              Download JSON
            </button>
          )}
        </div>

        {/* JSON Output */}
        {showJson && (
          <div>
            <h3 style={{ marginTop: '30px', color: '#333' }}>Generated Configuration:</h3>
            <pre style={jsonStyle}>
              {jsonOutput}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelConfigForm;