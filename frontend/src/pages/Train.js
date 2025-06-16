import React, { useState } from 'react';
import { Search, Plus } from 'lucide-react';
import NewModelTrainFlow from '../components/workflow/TrainModelFlow';
import TrainedModelSearch from '../components/search/TrainedModelSearch';

const Train = () => {
  const [activeOption, setActiveOption] = useState('search');
  const [selectedModel, setSelectedModel] = useState(null);

  const handleModelSelect = (model) => {
    setSelectedModel(model);
  };

  return (
    <div className="bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Model Training</h1>
          <p className="text-gray-600 mt-2">Search and manage existing trained models, or train a new one</p>
        </div>

        {/* Main Options */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <button
            onClick={() => setActiveOption('search')}
            className={`p-6 rounded-lg border-2 text-left transition-all ${
              activeOption === 'search'
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
            }`}
          >
            <div className="flex items-center gap-3 mb-3">
              <Search className="h-8 w-8 text-blue-500" />
              <h2 className="text-xl font-semibold text-gray-900">Search Existing Trained Models</h2>
            </div>
            <p className="text-gray-600">Browse and select from your existing trained models</p>
          </button>

          <button
            onClick={() => setActiveOption('create')}
            className={`p-6 rounded-lg border-2 text-left transition-all ${
              activeOption === 'create'
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
            }`}
          >
            <div className="flex items-center gap-3 mb-3">
              <Plus className="h-8 w-8 text-blue-500" />
              <h2 className="text-xl font-semibold text-gray-900">Train a New Model</h2>
            </div>
            <p className="text-gray-600">Configure and train a new model</p>
          </button>
          </div>

        {/* Content Area */}
        <div className="space-y-6">
          {activeOption === 'create' ? (
            <NewModelTrainFlow />
          ) : (
            <TrainedModelSearch 
            onModelSelect={handleModelSelect}
            selectedModel={selectedModel}
            showServingOptions={false}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default Train;