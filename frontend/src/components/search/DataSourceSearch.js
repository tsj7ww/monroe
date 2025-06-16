import React, { useState } from 'react';
import { Search, Database, FileText } from 'lucide-react';
import DataSourceDashboard from '../dashboards/DataSourceDashboard';

// Mock data for demonstration
const mockDataSources = {
    files: [
      {
        id: 'file_001',
        name: 'customer_sales_data.csv',
        type: 'CSV',
        size: '2.3 MB',
        rows: 15420,
        columns: 12,
        created: '2024-03-15',
        lastModified: '2024-03-20',
        owner: 'john.doe@company.com',
        tags: ['sales', 'customer', 'quarterly'],
        description: 'Quarterly customer sales data with transaction details',
        columns_info: ['customer_id', 'transaction_date', 'amount', 'product_category', 'region', 'sales_rep']
      },
      {
        id: 'file_002', 
        name: 'user_behavior_logs.json',
        type: 'JSON',
        size: '8.7 MB',
        rows: 45230,
        columns: 8,
        created: '2024-02-28',
        lastModified: '2024-03-18',
        owner: 'sarah.wilson@company.com',
        tags: ['analytics', 'user-behavior', 'web'],
        description: 'User interaction logs from web application',
        columns_info: ['user_id', 'session_id', 'action', 'timestamp', 'page_url', 'device_type']
      }
    ],
    databases: [
      {
        id: 'db_001',
        name: 'Production PostgreSQL',
        type: 'PostgreSQL',
        host: 'prod-db.company.com',
        database: 'main_analytics',
        tables: 23,
        created: '2024-01-10',
        lastAccessed: '2024-03-21',
        owner: 'db-admin@company.com',
        tags: ['production', 'analytics', 'real-time'],
        description: 'Main production database with customer and transaction data',
        connection_status: 'active'
      },
      {
        id: 'db_002',
        name: 'Marketing Data Warehouse',
        type: 'MySQL',
        host: 'marketing-dw.company.com', 
        database: 'marketing_analytics',
        tables: 15,
        created: '2024-02-05',
        lastAccessed: '2024-03-19',
        owner: 'marketing-team@company.com',
        tags: ['marketing', 'campaigns', 'historical'],
        description: 'Marketing campaign performance and customer segmentation data',
        connection_status: 'active'
      }
    ]
  };
  
  // Existing data source search component
  const DataSourceSearch = () => {
    const [activeTab, setActiveTab] = useState('files');
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedDataSource, setSelectedDataSource] = useState(null);
  
    const currentData = activeTab === 'files' ? mockDataSources.files : mockDataSources.databases;
    
    const filteredData = currentData.filter(item =>
      item.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.description.toLowerCase().includes(searchQuery.toLowerCase())
    );
  
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Search Existing Data Sources</h3>
          
          {/* Tab Navigation */}
          <div className="border-b border-gray-200 mb-6">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => {
                  setActiveTab('files');
                  setSelectedDataSource(null);
                }}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'files'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <FileText className="inline h-4 w-4 mr-2" />
                Files
              </button>
              <button
                disabled
                className={`py-2 px-1 border-b-2 font-medium text-sm border-gray-200 text-gray-400 cursor-not-allowed opacity-60`}
                // onClick={() => {
                //   setActiveTab('databases');
                //   setSelectedDataSource(null);
                // }}
                // className={`py-2 px-1 border-b-2 font-medium text-sm ${
                //   activeTab === 'databases'
                //     ? 'border-blue-500 text-blue-600'
                //     : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                // }`}
              >
                <Database className="inline h-4 w-4 mr-2" />
                Database Connections
              </button>
            </nav>
          </div>
  
          {/* Search Bar */}
          <div className="relative mb-6">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-4 w-4 text-gray-400" />
            </div>
            <input
              type="text"
              placeholder={`Search ${activeTab} by ID, name, or description...`}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="block w-full pl-9 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
  
          {/* Results List */}
          <div className="space-y-2">
            {filteredData.length > 0 ? (
              filteredData.map((item) => (
                <div
                  key={item.id}
                  onClick={() => setSelectedDataSource(item)}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    selectedDataSource?.id === item.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {activeTab === 'files' ? (
                        <FileText className="h-5 w-5 text-gray-500" />
                      ) : (
                        <Database className="h-5 w-5 text-gray-500" />
                      )}
                      <div>
                        <h4 className="font-medium text-gray-900">{item.name}</h4>
                        <p className="text-sm text-gray-600">ID: {item.id}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-gray-500">
                        {activeTab === 'files' ? item.type : item.type}
                      </span>
                      <p className="text-xs text-gray-400 mt-1">
                        {activeTab === 'files' ? item.size : `${item.tables} tables`}
                      </p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mt-2 line-clamp-2">{item.description}</p>
                </div>
              ))
            ) : (
              <div className="text-center py-8">
                <Search className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <p className="text-gray-500">No data sources found matching your search</p>
              </div>
            )}
          </div>
        </div>
  
        {/* Metadata Display */}
        <DataSourceDashboard dataSource={selectedDataSource} type={activeTab} />
      </div>
    );
  };

  export default DataSourceSearch;