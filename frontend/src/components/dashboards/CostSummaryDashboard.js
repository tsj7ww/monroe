import React, { useState } from 'react';
import { DollarSign, TrendingUp, TrendingDown, BarChart3, PieChart, Calendar, Download, ArrowUp, ArrowDown } from 'lucide-react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart as RePieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const CostDashboardDashboard = () => {
  const [timeRange, setTimeRange] = useState('6months');
  const [selectedService, setSelectedService] = useState('all');

  // Mock data
  const monthlySpending = [
    { month: 'Jan', spending: 3200, budget: 5000, models: 42 },
    { month: 'Feb', spending: 4100, budget: 5000, models: 58 },
    { month: 'Mar', spending: 3800, budget: 5000, models: 51 },
    { month: 'Apr', spending: 4500, budget: 5000, models: 67 },
    { month: 'May', spending: 4200, budget: 5000, models: 61 },
    { month: 'Jun', spending: 4832, budget: 5000, models: 73 }
  ];

  const serviceBreakdown = [
    { name: 'Model Training', value: 2100, percentage: 43, color: '#3B82F6' },
    { name: 'Model Serving', value: 1450, percentage: 30, color: '#10B981' },
    { name: 'Data Storage', value: 982, percentage: 20, color: '#8B5CF6' },
    { name: 'API Calls', value: 200, percentage: 4, color: '#F59E0B' },
    { name: 'Other', value: 100, percentage: 3, color: '#6B7280' }
  ];

  const topModels = [
    { name: 'Customer Churn Predictor', cost: 423, runs: 156, efficiency: 'High' },
    { name: 'Sales Forecast Model', cost: 356, runs: 98, efficiency: 'Medium' },
    { name: 'Fraud Detection v3', cost: 298, runs: 234, efficiency: 'High' },
    { name: 'Image Classifier', cost: 267, runs: 45, efficiency: 'Low' },
    { name: 'NLP Sentiment Analyzer', cost: 189, runs: 78, efficiency: 'Medium' }
  ];

  const costTrends = {
    currentMonth: 4832,
    lastMonth: 4200,
    monthChange: 15.0,
    yearToDate: 24630,
    lastYearToDate: 19800,
    yearChange: 24.4,
    avgMonthly: 4105,
    projectedAnnual: 49260
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Cost Dashboard</h2>
        <p className="mt-1 text-gray-600">Monitor your spending and resource utilization</p>
      </div>

      {/* Time Range Selector */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex gap-2">
          <button
            onClick={() => setTimeRange('1month')}
            className={`px-4 py-2 rounded-lg ${
              timeRange === '1month' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            1 Month
          </button>
          <button
            onClick={() => setTimeRange('3months')}
            className={`px-4 py-2 rounded-lg ${
              timeRange === '3months' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            3 Months
          </button>
          <button
            onClick={() => setTimeRange('6months')}
            className={`px-4 py-2 rounded-lg ${
              timeRange === '6months' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            6 Months
          </button>
          <button
            onClick={() => setTimeRange('1year')}
            className={`px-4 py-2 rounded-lg ${
              timeRange === '1year' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            1 Year
          </button>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
          <Download className="w-4 h-4" />
          Export Report
        </button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <DollarSign className="w-5 h-5 text-blue-600" />
            <span className={`flex items-center text-sm ${costTrends.monthChange > 0 ? 'text-red-600' : 'text-green-600'}`}>
              {costTrends.monthChange > 0 ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
              {Math.abs(costTrends.monthChange)}%
            </span>
          </div>
          <p className="text-sm text-gray-600">Current Month</p>
          <p className="text-2xl font-bold text-gray-900">${costTrends.currentMonth.toLocaleString()}</p>
          <p className="text-xs text-gray-500 mt-1">vs ${costTrends.lastMonth.toLocaleString()} last month</p>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-5 h-5 text-green-600" />
            <span className="text-sm text-green-600">
              <ArrowUp className="w-3 h-3 inline" />
              {costTrends.yearChange}%
            </span>
          </div>
          <p className="text-sm text-gray-600">Year to Date</p>
          <p className="text-2xl font-bold text-gray-900">${(costTrends.yearToDate / 1000).toFixed(1)}k</p>
          <p className="text-xs text-gray-500 mt-1">vs ${(costTrends.lastYearToDate / 1000).toFixed(1)}k last year</p>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <BarChart3 className="w-5 h-5 text-purple-600" />
          </div>
          <p className="text-sm text-gray-600">Average Monthly</p>
          <p className="text-2xl font-bold text-gray-900">${costTrends.avgMonthly.toLocaleString()}</p>
          <p className="text-xs text-gray-500 mt-1">Based on 6 months</p>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <Calendar className="w-5 h-5 text-orange-600" />
          </div>
          <p className="text-sm text-gray-600">Projected Annual</p>
          <p className="text-2xl font-bold text-gray-900">${(costTrends.projectedAnnual / 1000).toFixed(1)}k</p>
          <p className="text-xs text-gray-500 mt-1">At current rate</p>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Spending Trend Chart */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Spending Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={monthlySpending}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="month" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Legend />
              <Area 
                type="monotone" 
                dataKey="budget" 
                stroke="#e5e7eb" 
                fill="#f3f4f6" 
                name="Budget"
              />
              <Area 
                type="monotone" 
                dataKey="spending" 
                stroke="#3b82f6" 
                fill="#93bbfc" 
                name="Spending"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Service Breakdown */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Service Breakdown</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RePieChart>
              <Pie
                data={serviceBreakdown}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
              >
                {serviceBreakdown.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </RePieChart>
          </ResponsiveContainer>
          <div className="mt-4 space-y-2">
            {serviceBreakdown.map((service, index) => (
              <div key={index} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: service.color }} />
                  <span className="text-gray-700">{service.name}</span>
                </div>
                <span className="font-medium">${service.value} ({service.percentage}%)</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Additional Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Expensive Models */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Top Expensive Models</h3>
          <div className="space-y-3">
            {topModels.map((model, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex-1">
                  <p className="font-medium text-gray-900">{model.name}</p>
                  <p className="text-sm text-gray-500">{model.runs} runs • {model.efficiency} efficiency</p>
                </div>
                <div className="text-right">
                  <p className="font-semibold text-gray-900">${model.cost}</p>
                  <p className="text-xs text-gray-500">this month</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Cost Optimization Recommendations */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Cost Optimization Tips</h3>
          <div className="space-y-4">
            <div className="flex gap-3">
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-blue-600 font-bold">1</span>
              </div>
              <div>
                <p className="font-medium text-gray-900">Schedule training during off-peak hours</p>
                <p className="text-sm text-gray-500">Save up to 30% by running jobs between 12 AM - 6 AM</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-green-600 font-bold">2</span>
              </div>
              <div>
                <p className="font-medium text-gray-900">Enable auto-scaling for model serving</p>
                <p className="text-sm text-gray-500">Reduce serving costs by 25% during low-traffic periods</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-purple-600 font-bold">3</span>
              </div>
              <div>
                <p className="font-medium text-gray-900">Archive unused models and datasets</p>
                <p className="text-sm text-gray-500">Move to cold storage to save $150/month on storage costs</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-8 h-8 bg-orange-100 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-orange-600 font-bold">4</span>
              </div>
              <div>
                <p className="font-medium text-gray-900">Optimize model architectures</p>
                <p className="text-sm text-gray-500">Use efficient models to reduce training time by 40%</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CostDashboardDashboard;