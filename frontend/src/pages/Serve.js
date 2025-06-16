import React, { useState } from 'react';
import { Lock, Cloud, Download } from 'lucide-react';
import TrainedModelSearch from '../components/search/TrainedModelSearch';

const Serve = () => {
  const [selectedModel, setSelectedModel] = useState(null);
  const [showServingOptions, setShowServingOptions] = useState(false);

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    setShowServingOptions(false);
  };

  const handleServeModel = () => {
    setShowServingOptions(true);
  };

  const handleDownloadModel = () => {
    // Mock download functionality
    console.log(`Downloading model: ${selectedModel.name}`);
    alert(`Downloading ${selectedModel.name}.pkl (${selectedModel.modelSize})`);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Serving</h1>
        <p className="text-gray-600">Deploy your trained models for offline or online serving</p>
      </div>

      {/* Model Search Component */}
      <TrainedModelSearch 
        onModelSelect={handleModelSelect}
        selectedModel={selectedModel}
        showServingOptions={showServingOptions}
        onServeModel={handleServeModel}
        renderServingOptions={() => (
          <ServingOptions 
            selectedModel={selectedModel}
            onDownloadModel={handleDownloadModel}
          />
        )}
      />
    </div>
  );
};

// Serving Options Component
const ServingOptions = ({ selectedModel, onDownloadModel }) => {
  return (
    <div className="border-t border-gray-200 pt-6">
      <h4 className="text-lg font-semibold text-gray-900 mb-4">Serving Options</h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Offline Serving */}
        <div className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
          <div className="flex items-center mb-4">
            <Download className="w-8 h-8 text-blue-600 mr-3" />
            <h5 className="text-lg font-semibold text-gray-900">Offline Serving</h5>
          </div>
          <p className="text-gray-600 mb-4">
            Download the model as a pickle file for local deployment and inference.
          </p>
          <ul className="text-sm text-gray-500 mb-4 space-y-1">
            <li>• File format: .pkl</li>
            <li>• Size: {selectedModel.modelSize}</li>
            <li>• Python 3.8+ compatible</li>
          </ul>
          <button
            onClick={onDownloadModel}
            className="w-full bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors flex items-center justify-center"
          >
            <Download className="w-4 h-4 mr-2" />
            Download Model
          </button>
        </div>

        {/* Online Serving */}
        <div className="border border-gray-200 rounded-lg p-6 opacity-60">
          <div className="flex items-center mb-4">
            <Cloud className="w-8 h-8 text-gray-400 mr-3" />
            <h5 className="text-lg font-semibold text-gray-900">Online Serving</h5>
            <Lock className="w-5 h-5 text-gray-400 ml-auto" />
          </div>
          <p className="text-gray-600 mb-4">
            Deploy your model as a REST API endpoint for real-time predictions.
          </p>
          <ul className="text-sm text-gray-500 mb-4 space-y-1">
            <li>• Auto-scaling enabled</li>
            <li>• SSL/TLS secured</li>
            <li>• 99.9% uptime SLA</li>
          </ul>
          <button
            disabled
            className="w-full bg-gray-300 text-gray-500 px-4 py-2 rounded-md cursor-not-allowed flex items-center justify-center"
          >
            <Lock className="w-4 h-4 mr-2" />
            Coming Soon
          </button>
        </div>
      </div>
    </div>
  );
};

export default Serve;