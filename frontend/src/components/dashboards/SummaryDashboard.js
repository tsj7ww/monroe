import React, { useState, useEffect } from 'react';

const SummaryDashboard = () => {
    const [stats, setStats] = useState({
      totalModels: 0,
      activeTraining: 0,
      completedJobs: 0,
      accuracy: 0
    });
  
    const [loading, setLoading] = useState(true);
  
    // Simulate loading data
    useEffect(() => {
      const timer = setTimeout(() => {
        setStats({
          totalModels: 12,
          activeTraining: 2,
          completedJobs: 8,
          accuracy: 94.5
        });
        setLoading(false);
      }, 1000);
  
      return () => clearTimeout(timer);
    }, []);
  
    if (loading) {
      return (
        <div className="p-5 bg-gray-100">
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p>Loading...</p>
        </div>
      );
    }
  
    return (
      <div className="p-5 bg-gray-100">
        <h1 className="mb-8 text-3xl font-bold text-gray-800">
          ML Dashboard
        </h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
          <div className="bg-white p-5 rounded-lg shadow text-center">
            <h3 className="text-gray-600 mb-2">Total Models</h3>
            <div className="text-3xl font-bold text-blue-600">
              {stats.totalModels}
            </div>
          </div>
  
          <div className="bg-white p-5 rounded-lg shadow text-center">
            <h3 className="text-gray-600 mb-2">Active Training</h3>
            <div className="text-3xl font-bold text-yellow-500">
              {stats.activeTraining}
            </div>
          </div>
  
          <div className="bg-white p-5 rounded-lg shadow text-center">
            <h3 className="text-gray-600 mb-2">Completed Jobs</h3>
            <div className="text-3xl font-bold text-green-600">
              {stats.completedJobs}
            </div>
          </div>
  
          <div className="bg-white p-5 rounded-lg shadow text-center">
            <h3 className="text-gray-600 mb-2">Best Accuracy</h3>
            <div className="text-3xl font-bold text-red-600">
              {stats.accuracy}%
            </div>
          </div>
        </div>
  
        <div className="bg-white p-5 rounded-lg shadow">
          <h3 className="mb-4 text-xl font-semibold text-gray-800">Recent Activity</h3>
          <div className="space-y-0">
            <div className="p-2.5 border-b border-gray-200">
              ✅ Model training completed - Random Forest (94.5% accuracy)
            </div>
            <div className="p-2.5 border-b border-gray-200">
              🔄 Neural Network training in progress...
            </div>
            <div className="p-2.5 border-b border-gray-200">
              📊 New dataset uploaded - creditcard.csv
            </div>
            <div className="p-2.5">
              ⚙️ XGBoost model started training
            </div>
          </div>
        </div>
      </div>
    );
  };

export default SummaryDashboard;