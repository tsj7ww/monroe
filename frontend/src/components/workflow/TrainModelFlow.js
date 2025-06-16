import React, { useState } from 'react';
import ModelConfigForm from '../forms/ModelConfigForm';
import ModelConfigSearch from '../search/ModelConfigSearch';
import DataSourceForm from '../forms/DataSourceForm';
import DataSourceSearch from '../search/DataSourceSearch';
import { CheckCircle, Settings, Database, ChevronLeft, ChevronRight, Search, Play, Plus, RotateCcw } from 'lucide-react';

const ModelTrainFlow = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [config, setConfig] = useState({});
  const [data, setData] = useState({});
  const [showConfigSearch, setShowConfigSearch] = useState(false);
  const [showDataSearch, setShowDataSearch] = useState(false);
  const [showConfigForm, setShowConfigForm] = useState(false);
  const [showDataForm, setShowDataForm] = useState(false);

  const steps = [
      { id: 0, title: 'Model Configuration', icon: Settings },
      { id: 1, title: 'Data Selection', icon: Database },
      { id: 2, title: 'Summary & Review', icon: CheckCircle }
  ];
  
  const handleNext = () => {
      if (currentStep < steps.length - 1) {
          setCurrentStep(currentStep + 1);
      }
  };
  
  const handlePrev = () => {
      if (currentStep > 0) {
          setCurrentStep(currentStep - 1);
      }
  };

  const handleReset = () => {
      if (currentStep === 0) {
          // Reset model configuration
          setConfig({});
          setShowConfigSearch(false);
          setShowConfigForm(false);
      } else if (currentStep === 1) {
          // Reset data source
          setData({});
          setShowDataSearch(false);
          setShowDataForm(false);
      }
  };

  const handleConfigSelect = (selectedConfig) => {
      setConfig(selectedConfig);
      setShowConfigSearch(false);
  };

  const handleDataSelect = (selectedData) => {
      setData(selectedData);
      setShowDataSearch(false);
  };

  const resetConfigView = () => {
      setShowConfigSearch(false);
      setShowConfigForm(false);
  };

  const resetDataView = () => {
      setShowDataSearch(false);
      setShowDataForm(false);
  };

  return (
      <div className="min-h-screen bg-gray-50">
          <div className="container mx-auto p-6">
              {/* Header */}
              <div className="mb-8">
                  <h1 className="text-2xl font-bold text-gray-900 mb-2">Train a New Model</h1>
                  <p className="text-gray-600">Configure your model and data source to begin training</p>
              </div>

              {/* Progress Steps */}
              <div className="mb-8">
                  <div className="flex justify-between items-center">
                      {steps.map((step, index) => {
                          const Icon = step.icon;
                          const isActive = currentStep === index;
                          const isCompleted = currentStep > index;
                          
                          return (
                              <div key={step.id} className="flex items-center">
                                  <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all ${
                                      isCompleted 
                                          ? 'bg-green-500 border-green-500 text-white' 
                                          : isActive 
                                              ? 'border-blue-500 bg-blue-50 text-blue-600' 
                                              : 'border-gray-300 text-gray-400'
                                  }`}>
                                      {isCompleted ? (
                                          <CheckCircle className="w-6 h-6" />
                                      ) : (
                                          <Icon className="w-5 h-5" />
                                      )}
                                  </div>
                                  <div className={`ml-3 ${isActive ? 'text-blue-600' : isCompleted ? 'text-green-600' : 'text-gray-500'}`}>
                                      <div className="font-medium">{step.title}</div>
                                  </div>
                                  {index < steps.length - 1 && (
                                      <div className={`w-24 h-0.5 mx-4 ${isCompleted ? 'bg-green-500' : 'bg-gray-300'}`} />
                                  )}
                              </div>
                          );
                      })}
                  </div>
              </div>

              {/* Step Content */}
              <div className="mb-8">
                  {currentStep === 0 && (
                      <ModelConfigStep 
                          config={config}
                          setConfig={setConfig}
                          showSearch={showConfigSearch}
                          setShowSearch={setShowConfigSearch}
                          showForm={showConfigForm}
                          setShowForm={setShowConfigForm}
                          onConfigSelect={handleConfigSelect}
                          onReset={resetConfigView}
                      />
                  )}
                  {currentStep === 1 && (
                      <DataSourceStep 
                          data={data}
                          setData={setData}
                          showSearch={showDataSearch}
                          setShowSearch={setShowDataSearch}
                          showForm={showDataForm}
                          setShowForm={setShowDataForm}
                          onDataSelect={handleDataSelect}
                          onReset={resetDataView}
                      />
                  )}
                  {currentStep === 2 && <TrainSummary modelConfig={config} dataSource={data}/>}
              </div>

              {/* Navigation */}
              <div className="flex justify-between items-center">
                  <button
                      onClick={handlePrev}
                      disabled={currentStep === 0}
                      className={`flex items-center px-4 py-2 rounded-lg transition-colors ${
                          currentStep === 0 
                              ? 'text-gray-400 cursor-not-allowed' 
                              : 'text-gray-600 hover:bg-gray-100'
                      }`}
                  >
                      <ChevronLeft className="w-5 h-5 mr-1" />
                      Previous
                  </button>
                  
                  {/* Reset Button - only show for steps 0 and 1 */}
                  {currentStep < 2 && (
                      <button
                          onClick={handleReset}
                          className="flex items-center px-4 py-2 rounded-lg transition-colors text-red-600 hover:bg-red-50 border border-red-200"
                      >
                          <RotateCcw className="w-5 h-5 mr-1" />
                          {/* Reset {currentStep === 0 ? 'Configuration' : 'Data Source'} */}
                      </button>
                  )}
                  
                  <button
                      onClick={handleNext}
                      disabled={currentStep === steps.length - 1}
                      className={`flex items-center px-6 py-2 rounded-lg transition-colors ${
                          currentStep === steps.length - 1
                              ? 'text-gray-400 cursor-not-allowed'
                              : 'bg-blue-500 text-white hover:bg-blue-600'
                      }`}
                  >
                      Next
                      <ChevronRight className="w-5 h-5 ml-1" />
                  </button>
              </div>
          </div>
      </div>
  );
};

const ModelConfigStep = ({ 
  config, 
  setConfig, 
  showSearch, 
  setShowSearch, 
  showForm, 
  setShowForm, 
  onConfigSelect, 
  onReset 
}) => {
  if (showSearch) {
      return <ModelConfigSearch onSelect={onConfigSelect} onCancel={onReset} hideSelectButton={false}/>;
  }

  if (showForm) {
      return <ModelConfigForm config={config} setConfig={setConfig} onCancel={onReset} />;
  }

  return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="text-center py-12">
              <Settings className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Configure Your Model</h3>
              <p className="text-gray-600 mb-8">Choose an existing configuration or create a new one</p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button
                      onClick={() => setShowSearch(true)}
                      className="flex items-center justify-center px-6 py-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                      <Search className="w-5 h-5 mr-2" />
                      Search Existing Configs
                  </button>
                  <button
                      onClick={() => setShowForm(true)}
                      className="flex items-center justify-center px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                  >
                      <Plus className="w-5 h-5 mr-2" />
                      Create New Config
                  </button>
              </div>

              {Object.keys(config).length > 0 && (
                  <div className="mt-8 p-4 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex items-center text-green-800">
                          <CheckCircle className="w-5 h-5 mr-2" />
                          <span className="font-medium">Configuration Selected: {config.name || 'Unnamed Config'}</span>
                      </div>
                  </div>
              )}
          </div>
      </div>
  );
};

const DataSourceStep = ({ 
  data, 
  setData, 
  showSearch, 
  setShowSearch, 
  showForm, 
  setShowForm, 
  onDataSelect, 
  onReset 
}) => {
  if (showSearch) {
      return <DataSourceSearch onSelect={onDataSelect} onCancel={onReset} />;
  }

  if (showForm) {
      return <DataSourceForm data={data} setData={setData} onCancel={onReset} />;
  }

  return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="text-center py-12">
              <Database className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Select Your Data Source</h3>
              <p className="text-gray-600 mb-8">Choose an existing data source or configure a new one</p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button
                      onClick={() => setShowSearch(true)}
                      className="flex items-center justify-center px-6 py-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                      <Search className="w-5 h-5 mr-2" />
                      Search Existing Sources
                  </button>
                  <button
                      onClick={() => setShowForm(true)}
                      className="flex items-center justify-center px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                  >
                      <Plus className="w-5 h-5 mr-2" />
                      Create New Source
                  </button>
              </div>

              {Object.keys(data).length > 0 && (
                  <div className="mt-8 p-4 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex items-center text-green-800">
                          <CheckCircle className="w-5 h-5 mr-2" />
                          <span className="font-medium">Data Source Selected: {data.name || data.fileName || 'Unnamed Source'}</span>
                      </div>
                  </div>
              )}
          </div>
      </div>
  );
};

const TrainSummary = (modelConfig, dataSource) => (
    <div className="space-y-6">
      {/* Model Configuration Summary */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center mb-4">
          <Settings className="w-5 h-5 text-blue-600 mr-2" />
          <h3 className="text-lg font-semibold">Model Configuration</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">Model Name</div>
            <div className="font-medium">{modelConfig.name || 'Not specified'}</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">Model Type</div>
            <div className="font-medium">{modelConfig.type || 'Not specified'}</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">Learning Rate</div>
            <div className="font-medium">{modelConfig.learningRate || 'Not specified'}</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">Batch Size</div>
            <div className="font-medium">{modelConfig.batchSize || 'Not specified'}</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">Epochs</div>
            <div className="font-medium">{modelConfig.epochs || 'Not specified'}</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">Validation Split</div>
            <div className="font-medium">{modelConfig.validationSplit || 'Not specified'}</div>
          </div>
        </div>
        {modelConfig.description && (
          <div className="mt-4 bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">Description</div>
            <div className="font-medium">{modelConfig.description}</div>
          </div>
        )}
      </div>

      {/* Data Source Summary */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center mb-4">
          <Database className="w-5 h-5 text-green-600 mr-2" />
          <h3 className="text-lg font-semibold">Data Source</h3>
        </div>
        {dataSource.type === 'upload' ? (
          <div className="space-y-3">
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Source Type</div>
              <div className="font-medium">CSV Upload</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">File Name</div>
              <div className="font-medium">{dataSource.fileName || 'No file selected'}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">MinIO Storage Path</div>
              <div className="font-medium font-mono text-sm">{dataSource.minioPath || 'Not generated'}</div>
            </div>
          </div>
        ) : dataSource.type === 'database' ? (
          <div className="space-y-3">
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Source Type</div>
              <div className="font-medium">Database Connection</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Database Type</div>
              <div className="font-medium">{dataSource.config?.type || 'Not specified'}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Connection</div>
              <div className="font-medium">{dataSource.config?.host}:{dataSource.config?.port}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Query Preview</div>
              <div className="font-mono text-sm bg-gray-100 p-2 rounded mt-1">
                {dataSource.query?.substring(0, 100) || 'No query specified'}...
              </div>
            </div>
          </div>
        ) : (
          <div className="text-gray-500 italic">No data source configured</div>
        )}
      </div>

      {/* Training Action */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-semibold mb-2">Ready to Start Training</h3>
            <p className="text-blue-100">Review your configuration and start the training process</p>
          </div>
          <button className="bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors flex items-center">
            <Play className="w-5 h-5 mr-2" />
            Start Training
          </button>
        </div>
      </div>
    </div>
  );

export default ModelTrainFlow;