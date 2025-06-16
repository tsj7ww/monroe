import React, { useState } from 'react';
import AccuracyChart from '../charts/AccuracyChart';
import { Search, BarChart3, CheckCircle2, Loader2, AlertCircle, ChevronLeft, ChevronRight, Clock } from 'lucide-react';

// TrainedModelSearch Component
const TrainedModelSearch = ({ 
  onModelSelect, 
  selectedModel, 
  showServingOptions, 
  onServeModel,
  renderServingOptions,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [hasSearched, setHasSearched] = useState(false);

  // Mock data - replace with API call
  const mockModels = [
    {
      id: '1',
      name: 'Customer Churn Predictor',
      type: 'Classification',
      status: 'completed',
      accuracy: 0.94,
      createdAt: '2025-01-15',
      trainTime: '2h 15m',
      dataPoints: 50000,
      modelSize: '125 MB',
      lastServed: '2025-01-20'
    },
    {
      id: '2',
      name: 'Sales Forecast Model',
      type: 'Regression',
      status: 'training',
      accuracy: null,
      createdAt: '2025-01-20',
      trainTime: null,
      dataPoints: 75000,
      modelSize: null,
      lastServed: null
    },
    {
      id: '3',
      name: 'Anomaly Detector v2',
      type: 'Anomaly Detection',
      status: 'failed',
      accuracy: null,
      createdAt: '2025-01-18',
      trainTime: '45m',
      dataPoints: 30000,
      modelSize: null,
      lastServed: null
    },
    {
      id: '4',
      name: 'Time Series Forecasting',
      type: 'Regression',
      status: 'completed',
      accuracy: 0.88,
      createdAt: '2025-01-18',
      trainTime: '1h 30m',
      dataPoints: 30000,
      modelSize: '89 MB',
      lastServed: '2025-01-19'
    }
  ];

  const filteredModels = mockModels.filter(model =>
    model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    model.type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSearch = () => {
    if (searchTerm.trim()) {
      setHasSearched(true);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const handleModelClick = (model) => {
    onModelSelect(selectedModel?.id === model.id ? null : model);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-5 h-5 text-green-500" />;
      case 'training':
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return null;
    }
  };

  const getStatusBadge = (status) => {
    const statusClasses = {
      completed: 'bg-green-100 text-green-800',
      training: 'bg-blue-100 text-blue-800',
      failed: 'bg-red-100 text-red-800'
    };

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusClasses[status]}`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    );
  };

  return (
    <div>
      {/* Search Form */}
      <div className="mb-8">
        <div className="relative max-w-2xl">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            placeholder="Search by model name or type..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              if (e.target.value === '') {
                setHasSearched(false);
                onModelSelect(null);
              }
            }}
            onKeyPress={handleKeyPress}
            className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg"
          />
          <button
            onClick={handleSearch}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors"
          >
            Search
          </button>
        </div>
      </div>

      {/* Search Results */}
      {hasSearched && searchTerm && (
        <div className="space-y-6">
          <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 rounded-t-lg">
              <h3 className="font-semibold text-gray-900">Search Results</h3>
              <p className="text-sm text-gray-600">{filteredModels.length} model(s) found</p>
            </div>
            
            {filteredModels.length > 0 ? (
              <div className="divide-y divide-gray-200">
                {filteredModels.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => handleModelClick(model)}
                    className={`w-full px-4 py-4 hover:bg-gray-50 transition-colors text-left flex items-center justify-between ${
                      selectedModel?.id === model.id ? 'bg-blue-50 border-l-4 border-blue-500' : ''
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(model.status)}
                      <div>
                        <div className="font-medium text-gray-900">{model.name}</div>
                        <div className="text-sm text-gray-500">{model.type}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      {getStatusBadge(model.status)}
                      <ChevronRight className={`w-5 h-5 text-gray-400 transition-transform ${
                        selectedModel?.id === model.id ? 'rotate-90' : ''
                      }`} />
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="px-4 py-8 text-center text-gray-500">
                No models found matching "{searchTerm}"
              </div>
            )}
          </div>

          {/* Model Details Section */}
          {selectedModel && (
            <ModelDetails 
              model={selectedModel}
              showServingOptions={showServingOptions}
              onServeModel={onServeModel}
              onClose={() => onModelSelect(null)}
              renderServingOptions={renderServingOptions}
              getStatusIcon={getStatusIcon}
              getStatusBadge={getStatusBadge}
            />
          )}
        </div>
      )}
    </div>
  );
};

// Model Details Component
const ModelDetails = ({ 
  model, 
  showServingOptions, 
  onServeModel, 
  onClose,
  renderServingOptions,
  getStatusIcon,
  getStatusBadge
}) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
      <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-gray-900">{model.name}</h3>
            <p className="text-gray-500">{model.type}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 p-1"
          >
            <ChevronLeft className="w-5 h-5" />
            <span className="sr-only">Hide details</span>
          </button>
        </div>
      </div>

      <div className="p-6">
        {/* Performance Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg">
            <div className="flex items-center mb-3">
              <BarChart3 className="w-6 h-6 text-blue-600 mr-3" />
              <span className="text-sm font-medium text-blue-800">Model Accuracy</span>
            </div>
            <p className="text-3xl font-bold text-blue-900">
              {model.accuracy ? `${(model.accuracy * 100).toFixed(1)}%` : 'N/A'}
            </p>
            {model.status === 'training' && (
              <p className="text-sm text-blue-600 mt-1">Training in progress...</p>
            )}
          </div>

          <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-lg">
            <div className="flex items-center mb-3">
              <Clock className="w-6 h-6 text-green-600 mr-3" />
              <span className="text-sm font-medium text-green-800">Training Time</span>
            </div>
            <p className="text-3xl font-bold text-green-900">
              {model.trainTime || 'In Progress'}
            </p>
            {model.status === 'failed' && (
              <p className="text-sm text-red-600 mt-1">Training failed</p>
            )}
          </div>
        </div>

        {/* Detailed Information */}
        <div className="space-y-6">
          <div>
            <h4 className="text-lg font-semibold text-gray-900 mb-4">Model Information</h4>
            <div className="bg-gray-50 p-6 rounded-lg">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">Status</p>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(model.status)}
                    {getStatusBadge(model.status)}
                  </div>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">Data Points</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {model.dataPoints.toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">Created Date</p>
                  <p className="text-lg font-semibold text-gray-900">{model.createdAt}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">Model Size</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {model.modelSize || 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">Last Served</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {model.lastServed || 'Never'}
                  </p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">Model ID</p>
                  <p className="text-sm font-mono bg-white px-3 py-2 rounded border">
                    {model.id}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Accuracy Chart - Only show for completed models */}
          {model.status === 'completed' && (
            <div className="mb-6">
              <AccuracyChart />
            </div>
          )}

          {/* Serving Options */}
          {showServingOptions && model.status === 'completed' && renderServingOptions()}

          {/* Action Buttons */}
          <div className="flex space-x-3 pt-4 border-t border-gray-200">
            {model.status === 'completed' && showServingOptions && (
                <button 
                  onClick={onServeModel}
                  className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors"
                >
                  Serve Model
                </button>
            )}
            {model.status === 'completed' && (
                <button className="bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600 transition-colors">
                  View Full Report
                </button>
            )}
            {model.status === 'training' && (
              <button className="bg-orange-500 text-white px-4 py-2 rounded-md hover:bg-orange-600 transition-colors">
                View Training Progress
              </button>
            )}
            {model.status === 'failed' && (
              <button className="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600 transition-colors">
                View Error Log
              </button>
            )}
            <button className="border border-gray-300 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-50 transition-colors">
              Export Details
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainedModelSearch;