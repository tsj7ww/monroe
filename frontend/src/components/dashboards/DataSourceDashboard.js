import React from 'react';
import { Database, FileText, HardDrive, Calendar, User, Tag, BarChart3 } from 'lucide-react';

const DataSourceDashboard = ({ dataSource, type }) => {
    if (!dataSource) return null;
  
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6 mt-4">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              {type === 'files' ? <FileText className="h-5 w-5" /> : <Database className="h-5 w-5" />}
              {dataSource.name}
            </h3>
            <p className="text-gray-600 mt-1">{dataSource.description}</p>
          </div>
          <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
            {type === 'files' ? dataSource.type : dataSource.connection_status}
          </span>
        </div>
  
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {type === 'files' ? (
            <>
              <div className="flex items-center gap-2">
                <HardDrive className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-600">Size: {dataSource.size}</span>
              </div>
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-600">Rows: {dataSource.rows?.toLocaleString()}</span>
              </div>
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-600">Columns: {dataSource.columns}</span>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-600">Host: {dataSource.host}</span>
              </div>
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-600">Tables: {dataSource.tables}</span>
              </div>
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-600">Database: {dataSource.database}</span>
              </div>
            </>
          )}
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-gray-500" />
            <span className="text-sm text-gray-600">Created: {dataSource.created}</span>
          </div>
          <div className="flex items-center gap-2">
            <User className="h-4 w-4 text-gray-500" />
            <span className="text-sm text-gray-600">Owner: {dataSource.owner}</span>
          </div>
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-gray-500" />
            <span className="text-sm text-gray-600">
              {type === 'files' ? `Modified: ${dataSource.lastModified}` : `Accessed: ${dataSource.lastAccessed}`}
            </span>
          </div>
        </div>
  
        {dataSource.tags && (
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <Tag className="h-4 w-4 text-gray-500" />
              <span className="text-sm text-gray-600">Tags:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {dataSource.tags.map((tag, index) => (
                <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
  
        {type === 'files' && dataSource.columns_info && (
          <div>
            <h4 className="text-sm font-medium text-gray-900 mb-2">Column Information</h4>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {dataSource.columns_info.map((column, index) => (
                  <span key={index} className="text-xs bg-white px-2 py-1 rounded border">
                    {column}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  export default DataSourceDashboard;