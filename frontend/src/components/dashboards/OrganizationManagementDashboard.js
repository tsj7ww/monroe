import React, { useState } from 'react';
import { Building2, Globe, Users, Shield, Key, AlertCircle, MoreVertical } from 'lucide-react';

const OrganizationManagementDashboard = () => {
  const [orgData, setOrgData] = useState({
    name: 'Acme Corporation',
    industry: 'Technology',
    size: '51-200 employees',
    website: 'https://acme.com',
    logo: null
  });

  const [settings, setSettings] = useState({
    allowProjectCreation: true,
    requireDeploymentApproval: false,
    enableSSO: false,
    enforce2FA: true,
    dataRetentionDays: 90,
    allowExternalSharing: false
  });

  const [teams, setTeams] = useState([
    { id: 1, name: 'Data Science', members: 8, projects: 12 },
    { id: 2, name: 'Engineering', members: 15, projects: 23 },
    { id: 3, name: 'Research', members: 6, projects: 9 }
  ]);

  const handleSettingChange = (setting) => {
    setSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }));
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Organization Management</h2>
        <p className="mt-1 text-gray-600">Configure your organization settings and preferences</p>
      </div>

      <div className="space-y-6">
        {/* Organization Profile */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Organization Profile</h3>
            <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
              Edit Profile
            </button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Organization Name</label>
                <input 
                  type="text" 
                  value={orgData.name}
                  onChange={(e) => setOrgData({...orgData, name: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Industry</label>
                <select 
                  value={orgData.industry}
                  onChange={(e) => setOrgData({...orgData, industry: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option>Technology</option>
                  <option>Healthcare</option>
                  <option>Finance</option>
                  <option>Retail</option>
                  <option>Manufacturing</option>
                  <option>Education</option>
                </select>
              </div>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Company Size</label>
                <select 
                  value={orgData.size}
                  onChange={(e) => setOrgData({...orgData, size: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option>1-10 employees</option>
                  <option>11-50 employees</option>
                  <option>51-200 employees</option>
                  <option>201-500 employees</option>
                  <option>501+ employees</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Website</label>
                <input 
                  type="url" 
                  value={orgData.website}
                  onChange={(e) => setOrgData({...orgData, website: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>

          {/* Logo Upload */}
          <div className="mt-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">Organization Logo</label>
            <div className="flex items-center gap-4">
              <div className="w-20 h-20 bg-gray-200 rounded-lg flex items-center justify-center">
                <Building2 className="w-8 h-8 text-gray-400" />
              </div>
              <button className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
                Upload Logo
              </button>
            </div>
          </div>
        </div>

        {/* Security & Permissions */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Shield className="w-5 h-5 text-gray-700" />
            <h3 className="text-lg font-medium text-gray-900">Security & Permissions</h3>
          </div>
          
          <div className="space-y-4">
            <SettingToggle
              icon={<Users className="w-5 h-5" />}
              title="Allow team members to create projects"
              description="Team members can create and manage their own ML projects"
              enabled={settings.allowProjectCreation}
              onChange={() => handleSettingChange('allowProjectCreation')}
            />
            <SettingToggle
              icon={<Shield className="w-5 h-5" />}
              title="Require approval for model deployment"
              description="Models must be approved before being deployed to production"
              enabled={settings.requireDeploymentApproval}
              onChange={() => handleSettingChange('requireDeploymentApproval')}
            />
            <SettingToggle
              icon={<Key className="w-5 h-5" />}
              title="Enable SSO authentication"
              description="Use single sign-on for all team members"
              enabled={settings.enableSSO}
              onChange={() => handleSettingChange('enableSSO')}
            />
            <SettingToggle
              icon={<AlertCircle className="w-5 h-5" />}
              title="Enforce two-factor authentication"
              description="Require 2FA for all users in the organization"
              enabled={settings.enforce2FA}
              onChange={() => handleSettingChange('enforce2FA')}
            />
          </div>
        </div>

        {/* Data & Privacy */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Data & Privacy</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Data Retention Period</label>
              <select 
                value={settings.dataRetentionDays}
                onChange={(e) => setSettings({...settings, dataRetentionDays: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={30}>30 days</option>
                <option value={60}>60 days</option>
                <option value={90}>90 days</option>
                <option value={180}>180 days</option>
                <option value={365}>1 year</option>
              </select>
            </div>
            <SettingToggle
              icon={<Globe className="w-5 h-5" />}
              title="Allow external data sharing"
              description="Allow sharing of models and data with users outside the organization"
              enabled={settings.allowExternalSharing}
              onChange={() => handleSettingChange('allowExternalSharing')}
            />
          </div>
        </div>

        {/* Teams */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Teams</h3>
            <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
              + Create Team
            </button>
          </div>
          
          <div className="space-y-3">
            {teams.map((team) => (
              <div key={team.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                    <Users className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{team.name}</p>
                    <p className="text-sm text-gray-500">{team.members} members • {team.projects} projects</p>
                  </div>
                </div>
                <button className="text-gray-400 hover:text-gray-600">
                  <MoreVertical className="w-5 h-5" />
                </button>
              </div>
            ))}
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

// Setting Toggle Component
const SettingToggle = ({ icon, title, description, enabled, onChange }) => {
  return (
    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
      <div className="flex items-start gap-3">
        <div className="text-gray-600 mt-0.5">{icon}</div>
        <div className="flex-1">
          <p className="font-medium text-gray-900">{title}</p>
          <p className="text-sm text-gray-500 mt-0.5">{description}</p>
        </div>
      </div>
      <button
        onClick={onChange}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
          enabled ? 'bg-blue-600' : 'bg-gray-300'
        }`}
      >
        <span
          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
            enabled ? 'translate-x-6' : 'translate-x-1'
          }`}
        />
      </button>
    </div>
  );
};

export default OrganizationManagementDashboard;