import React, { useState } from 'react';
import DataSelectionForm from '../components/forms/DataSelectionForm';
import ExistingDataSourceSearch from '../components/search/DataSources';
import { Search, Plus } from 'lucide-react';

// Main Data Page Component
const Data = () => {
  const [activeOption, setActiveOption] = useState('search');

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Data Management</h1>
          <p className="text-gray-600 mt-2">Manage your data sources and connections for automated machine learning</p>
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
              <h2 className="text-xl font-semibold text-gray-900">Search Existing Data Sources</h2>
            </div>
            <p className="text-gray-600">Browse and select from your existing files and database connections</p>
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
              <h2 className="text-xl font-semibold text-gray-900">Create New Data Source</h2>
            </div>
            <p className="text-gray-600">Add a new file upload or database connection to your AutoML workflow</p>
          </button>
          </div>

        {/* Content Area */}
        <div className="space-y-6">
          {activeOption === 'create' ? (
            <DataSelectionForm />
          ) : (
            <ExistingDataSourceSearch />
          )}
        </div>
      </div>
    </div>
  );
};

export default Data;