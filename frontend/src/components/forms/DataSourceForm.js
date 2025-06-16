import React, { useState } from 'react';
import { Upload, Database, CheckCircle } from 'lucide-react';

const DataSourceForm = () => {
    const [dataSourceType, setDataSourceType] = useState('upload');
    const [dbConfig, setDbConfig] = useState({});
    const [sqlQuery, setSqlQuery] = useState('SELECT * FROM your_table LIMIT 1000;');
    const [dataSource, setDataSource] = useState({});
    const [uploadedFile, setUploadedFile] = useState(null);

    const handleFileUpload = (event) => {
      const file = event.target.files[0];
      setUploadedFile(file);
      setDataSource({
        type: 'upload',
        fileName: file?.name,
        size: file?.size,
        minioPath: `uploads/${file?.name}-${Date.now()}`
      });
    };

    return (
      <div className="space-y-6">
        {/* Data Source Selection */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold mb-4">Choose Data Source</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <button
              onClick={() => setDataSourceType('upload')}
              className={`p-4 border-2 rounded-lg transition-all ${
                dataSourceType === 'upload' 
                  ? 'border-blue-500 bg-blue-50 text-blue-700' 
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <Upload className="w-8 h-8 mx-auto mb-2" />
              <div className="font-medium">Upload CSV</div>
              <div className="text-sm text-gray-500">Upload a CSV file from your computer</div>
            </button>
            <button
              disabled
              className={"p-4 border-2 rounded-lg border-gray-200 bg-gray-100 text-gray-400 cursor-not-allowed opacity-60"}
              // onClick={() => setDataSourceType('database')}
              // className={`p-4 border-2 rounded-lg transition-all ${
              //   dataSourceType === 'database' 
              //     ? 'border-blue-500 bg-blue-50 text-blue-700' 
              //     : 'border-gray-200 hover:border-gray-300'
              // }`}
            >
              <Database className="w-8 h-8 mx-auto mb-2" />
              <div className="font-medium">Database Connection</div>
              <div className="text-sm text-gray-400">Connect to an external database (Coming Soon)</div>
            </button>
          </div>
        </div>

        {/* CSV Upload Section */}
        {dataSourceType === 'upload' && (
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-semibold mb-4">Upload CSV File</h3>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors">
              <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <div className="mb-4">
                <label htmlFor="file-upload" className="cursor-pointer">
                  <span className="text-blue-600 hover:text-blue-500 font-medium">Choose a file</span>
                  <span className="text-gray-500"> or drag and drop</span>
                </label>
                <input
                  id="file-upload"
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleFileUpload}
                />
              </div>
              <p className="text-sm text-gray-500">CSV files up to 100MB</p>
            </div>
            {uploadedFile && (
              <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center">
                  <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
                  <div>
                    <div className="font-medium text-green-800">{uploadedFile.name}</div>
                    <div className="text-sm text-green-600">
                      {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB • Ready to upload to MinIO
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Database Connection Section */}
        {dataSourceType === 'database' && (
          <div className="bg-white rounded-lg shadow-sm border p-6 space-y-6">
            <h3 className="text-lg font-semibold">Database Connection</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Database Type</label>
                <select 
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  onChange={(e) => setDbConfig({...dbConfig, type: e.target.value})}
                >
                  <option value="">Select database type</option>
                  <option value="postgresql">PostgreSQL</option>
                  <option value="mysql">MySQL</option>
                  <option value="sqlite">SQLite</option>
                  <option value="mongodb">MongoDB</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Host</label>
                <input 
                  type="text" 
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="localhost"
                  onChange={(e) => setDbConfig({...dbConfig, host: e.target.value})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Port</label>
                <input 
                  type="number" 
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="5432"
                  onChange={(e) => setDbConfig({...dbConfig, port: e.target.value})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Database Name</label>
                <input 
                  type="text" 
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="database_name"
                  onChange={(e) => setDbConfig({...dbConfig, database: e.target.value})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Username</label>
                <input 
                  type="text" 
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="username"
                  onChange={(e) => setDbConfig({...dbConfig, username: e.target.value})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                <input 
                  type="password" 
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="password"
                  onChange={(e) => setDbConfig({...dbConfig, password: e.target.value})}
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">SQL Query</label>
              <div className="relative">
                <textarea
                  rows="6"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
                  value={sqlQuery}
                  onChange={(e) => setSqlQuery(e.target.value)}
                />
                <button className="absolute top-2 right-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors">
                  Test Query
                </button>
              </div>
              <p className="text-sm text-gray-500 mt-1">Write your SQL query to fetch training data</p>
            </div>

            <button
              onClick={() => setDataSource({
                type: 'database',
                config: dbConfig,
                query: sqlQuery
              })}
              className="w-full px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors"
            >
              Test Connection
            </button>
          </div>
        )}
      </div>
    );
  };

  export default DataSourceForm;