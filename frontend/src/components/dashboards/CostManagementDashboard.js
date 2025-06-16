import React, { useState } from 'react';
import { DollarSign, AlertTriangle, Bell, CreditCard, TrendingUp, Settings, Mail, Smartphone } from 'lucide-react';

const CostManagementDashboard = () => {
  const [budget, setBudget] = useState({
    monthlyLimit: 5000,
    alertThreshold: 75,
    autoShutoff: true,
    shutoffThreshold: 100
  });

  const [alerts, setAlerts] = useState({
    email: true,
    sms: false,
    dailySummary: true,
    weeklyReport: true,
    thresholdAlert: true,
    unusualSpending: true
  });

  const [billingInfo, setBillingInfo] = useState({
    paymentMethod: '•••• 4242',
    billingCycle: 'Monthly',
    nextBillingDate: 'February 1, 2025',
    billingEmail: 'billing@acme.com'
  });

  const [costAllocations, setCostAllocations] = useState([
    { team: 'Data Science', budget: 2000, spent: 1650, percentage: 82.5 },
    { team: 'Engineering', budget: 2000, spent: 1890, percentage: 94.5 },
    { team: 'Research', budget: 1000, spent: 450, percentage: 45 }
  ]);

  return (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Cost Management</h2>
        <p className="mt-1 text-gray-600">Set budgets and configure billing alerts</p>
      </div>

      <div className="space-y-6">
        {/* Budget Overview */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Current Month Overview</h3>
          <div className="mb-6">
            <div className="flex justify-between mb-2">
              <span className="text-sm text-gray-600">Monthly Budget Usage</span>
              <span className="text-sm font-medium">$4,832 / $5,000</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div className="bg-orange-500 h-3 rounded-full" style={{ width: '96.6%' }} />
            </div>
            <p className="text-xs text-orange-600 mt-1">Warning: 96.6% of budget consumed</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <TrendingUp className="w-5 h-5 text-blue-600" />
                <span className="text-xs text-blue-600">+15%</span>
              </div>
              <p className="text-sm text-blue-600">Daily Average</p>
              <p className="text-xl font-bold text-blue-900">$161</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <p className="text-sm text-green-600 mb-1">Remaining Budget</p>
              <p className="text-xl font-bold text-green-900">$168</p>
              <p className="text-xs text-green-600">3 days left</p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <p className="text-sm text-purple-600 mb-1">Projected Total</p>
              <p className="text-xl font-bold text-purple-900">$5,123</p>
              <p className="text-xs text-red-600">$123 over budget</p>
            </div>
          </div>
        </div>

        {/* Budget Settings */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Budget Settings</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Monthly Budget Limit</label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500">$</span>
                <input 
                  type="number" 
                  value={budget.monthlyLimit}
                  onChange={(e) => setBudget({...budget, monthlyLimit: parseInt(e.target.value)})}
                  className="w-full pl-8 pr-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Alert Threshold</label>
              <div className="flex items-center gap-2">
                <input 
                  type="range" 
                  min="50" 
                  max="100" 
                  value={budget.alertThreshold}
                  onChange={(e) => setBudget({...budget, alertThreshold: parseInt(e.target.value)})}
                  className="flex-1"
                />
                <span className="w-12 text-sm font-medium">{budget.alertThreshold}%</span>
              </div>
            </div>
            <div className="md:col-span-2">
              <label className="flex items-center gap-3">
                <input 
                  type="checkbox" 
                  checked={budget.autoShutoff}
                  onChange={(e) => setBudget({...budget, autoShutoff: e.target.checked})}
                  className="w-5 h-5 text-blue-600 rounded"
                />
                <div>
                  <p className="font-medium text-gray-700">Enable automatic shutoff</p>
                  <p className="text-sm text-gray-500">Automatically pause all training when budget is exceeded</p>
                </div>
              </label>
              {budget.autoShutoff && (
                <div className="ml-8 mt-3">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Shutoff Threshold</label>
                  <select 
                    value={budget.shutoffThreshold}
                    onChange={(e) => setBudget({...budget, shutoffThreshold: parseInt(e.target.value)})}
                    className="w-48 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value={100}>100% of budget</option>
                    <option value={110}>110% of budget</option>
                    <option value={120}>120% of budget</option>
                  </select>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Alert Configuration */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Alert Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <AlertOption
              icon={<Mail className="w-5 h-5" />}
              title="Email Alerts"
              description="Receive alerts via email"
              enabled={alerts.email}
              onChange={(e) => setAlerts({...alerts, email: e.target.checked})}
            />
            <AlertOption
              icon={<Smartphone className="w-5 h-5" />}
              title="SMS Alerts"
              description="Receive alerts via SMS"
              enabled={alerts.sms}
              onChange={(e) => setAlerts({...alerts, sms: e.target.checked})}
            />
            <AlertOption
              icon={<Bell className="w-5 h-5" />}
              title="Daily Summary"
              description="Daily spending report"
              enabled={alerts.dailySummary}
              onChange={(e) => setAlerts({...alerts, dailySummary: e.target.checked})}
            />
            <AlertOption
              icon={<TrendingUp className="w-5 h-5" />}
              title="Weekly Report"
              description="Weekly cost analysis"
              enabled={alerts.weeklyReport}
              onChange={(e) => setAlerts({...alerts, weeklyReport: e.target.checked})}
            />
            <AlertOption
              icon={<AlertTriangle className="w-5 h-5" />}
              title="Threshold Alerts"
              description="Alert when threshold reached"
              enabled={alerts.thresholdAlert}
              onChange={(e) => setAlerts({...alerts, thresholdAlert: e.target.checked})}
            />
            <AlertOption
              icon={<DollarSign className="w-5 h-5" />}
              title="Unusual Spending"
              description="Alert on spending spikes"
              enabled={alerts.unusualSpending}
              onChange={(e) => setAlerts({...alerts, unusualSpending: e.target.checked})}
            />
          </div>
        </div>

        {/* Team Budget Allocation */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Team Budget Allocation</h3>
          <div className="space-y-4">
            {costAllocations.map((allocation, index) => (
              <div key={index} className="border-b border-gray-200 pb-4 last:border-0">
                <div className="flex justify-between mb-2">
                  <span className="font-medium text-gray-900">{allocation.team}</span>
                  <span className="text-sm text-gray-600">
                    ${allocation.spent} / ${allocation.budget}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${
                      allocation.percentage > 90 ? 'bg-red-500' : 
                      allocation.percentage > 75 ? 'bg-orange-500' : 'bg-green-500'
                    }`} 
                    style={{ width: `${allocation.percentage}%` }} 
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">{allocation.percentage}% utilized</p>
              </div>
            ))}
          </div>
        </div>

        {/* Billing Information */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Billing Information</h3>
          <div className="bg-gray-50 p-4 rounded-lg mb-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Payment Method</span>
                <span className="font-medium flex items-center gap-2">
                  <CreditCard className="w-4 h-4" />
                  {billingInfo.paymentMethod}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Billing Cycle</span>
                <span className="font-medium">{billingInfo.billingCycle}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Next Billing Date</span>
                <span className="font-medium">{billingInfo.nextBillingDate}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Billing Email</span>
                <span className="font-medium">{billingInfo.billingEmail}</span>
              </div>
            </div>
          </div>
          <div className="flex gap-3">
            <button className="text-blue-600 hover:text-blue-800 font-medium">
              Update Payment Method
            </button>
            <button className="text-blue-600 hover:text-blue-800 font-medium">
              Download Invoices
            </button>
          </div>
        </div>

        {/* Save Button */}
        <div className="flex justify-end gap-3">
          <button className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
            Cancel
          </button>
          <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
};

// Alert Option Component
const AlertOption = ({ icon, title, description, enabled, onChange }) => {
  return (
    <label className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100">
      <input 
        type="checkbox" 
        checked={enabled}
        onChange={onChange}
        className="w-5 h-5 text-blue-600 rounded"
      />
      <div className="flex-1">
        <div className="flex items-center gap-2 text-gray-700">
          {icon}
          <span className="font-medium">{title}</span>
        </div>
        <p className="text-sm text-gray-500 mt-0.5">{description}</p>
      </div>
    </label>
  );
};

export default CostManagementDashboard;