import React, { useState } from 'react';
import { User, Building2, CreditCard, BarChart3, ChevronRight } from 'lucide-react';
import UserManagementDashboard from '../components/dashboards/UserManagementDashboard';
import OrganizationManagementDashboard from '../components/dashboards/OrganizationManagementDashboard';
import CostManagementDashboard from '../components/dashboards/CostManagementDashboard';
import CostDashboardDashboard from '../components/dashboards/CostSummaryDashboard';

const Settings = () => {
  const [activeSection, setActiveSection] = useState('user');

  const sections = [
    {
      id: 'user',
      title: 'User Management',
      icon: User,
      description: 'Manage user profiles, permissions, and access controls'
    },
    {
      id: 'organization',
      title: 'Organization Management',
      icon: Building2,
      description: 'Configure organization settings, teams, and policies'
    },
    {
      id: 'cost',
      title: 'Cost Management',
      icon: CreditCard,
      description: 'Set budgets, configure alerts, and manage billing'
    },
    {
      id: 'cost-dashboard',
      title: 'Cost Dashboard',
      icon: BarChart3,
      description: 'View spending analytics and resource utilization'
    }
  ];

  const renderContent = () => {
    switch (activeSection) {
      case 'user':
        return <UserManagementDashboard />;
      case 'organization':
        return <OrganizationManagementDashboard />;
      case 'cost':
        return <CostManagementDashboard />;
      case 'cost-dashboard':
        return <CostDashboardDashboard />;
      default:
        return <UserManagementDashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <p className="mt-2 text-gray-600">Manage your account, organization, and billing preferences</p>
        </div>

        <div className="flex flex-col lg:flex-row gap-8">
          {/* Sidebar Navigation */}
          <div className="lg:w-1/4">
            <nav className="space-y-2">
              {sections.map((section) => {
                const Icon = section.icon;
                return (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-all duration-200 ${
                      activeSection === section.id
                        ? 'bg-blue-50 border-l-4 border-blue-500 text-blue-700'
                        : 'hover:bg-gray-100 text-gray-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <Icon className={`w-5 h-5 ${
                          activeSection === section.id ? 'text-blue-600' : 'text-gray-400'
                        }`} />
                        <div>
                          <p className="font-medium">{section.title}</p>
                          <p className="text-xs text-gray-500 mt-1">{section.description}</p>
                        </div>
                      </div>
                      {activeSection === section.id && (
                        <ChevronRight className="w-4 h-4 text-blue-600" />
                      )}
                    </div>
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Main Content Area */}
          <div className="lg:w-3/4">
            <div className="bg-white rounded-lg shadow">
              {renderContent()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;