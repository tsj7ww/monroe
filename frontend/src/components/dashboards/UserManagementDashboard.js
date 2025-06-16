import React, { useState } from 'react';
import { User, Mail, Shield, MoreVertical, Search, Filter, UserPlus, Download } from 'lucide-react';

const UserManagementDashboard = () => {
  const [users, setUsers] = useState([
    { id: 1, name: 'John Doe', email: 'john@example.com', role: 'Admin', status: 'Active', lastLogin: '2025-01-20 14:30', avatar: 'JD' },
    { id: 2, name: 'Jane Smith', email: 'jane@example.com', role: 'Developer', status: 'Active', lastLogin: '2025-01-20 09:15', avatar: 'JS' },
    { id: 3, name: 'Bob Wilson', email: 'bob@example.com', role: 'Viewer', status: 'Inactive', lastLogin: '2025-01-15 16:45', avatar: 'BW' },
    { id: 4, name: 'Sarah Chen', email: 'sarah@example.com', role: 'Developer', status: 'Active', lastLogin: '2025-01-19 11:20', avatar: 'SC' },
    { id: 5, name: 'Mike Johnson', email: 'mike@example.com', role: 'Viewer', status: 'Active', lastLogin: '2025-01-20 13:00', avatar: 'MJ' }
  ]);

  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRole, setSelectedRole] = useState('all');
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDropdown, setShowDropdown] = useState(null);

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRole = selectedRole === 'all' || user.role === selectedRole;
    return matchesSearch && matchesRole;
  });

  const getRoleBadgeColor = (role) => {
    switch (role) {
      case 'Admin':
        return 'bg-purple-100 text-purple-800';
      case 'Developer':
        return 'bg-blue-100 text-blue-800';
      case 'Viewer':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusBadgeColor = (status) => {
    return status === 'Active' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">User Management</h2>
        <p className="mt-1 text-gray-600">Add, remove, and manage user permissions</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600">Total Users</p>
              <p className="text-2xl font-bold text-blue-900">{users.length}</p>
            </div>
            <User className="w-8 h-8 text-blue-500" />
          </div>
        </div>
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-green-600">Active Users</p>
              <p className="text-2xl font-bold text-green-900">
                {users.filter(u => u.status === 'Active').length}
              </p>
            </div>
            <div className="w-8 h-8 bg-green-500 rounded-full" />
          </div>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-purple-600">Admins</p>
              <p className="text-2xl font-bold text-purple-900">
                {users.filter(u => u.role === 'Admin').length}
              </p>
            </div>
            <Shield className="w-8 h-8 text-purple-500" />
          </div>
        </div>
        <div className="bg-gray-50 p-4 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Developers</p>
              <p className="text-2xl font-bold text-gray-900">
                {users.filter(u => u.role === 'Developer').length}
              </p>
            </div>
            <div className="w-8 h-8 bg-gray-400 rounded" />
          </div>
        </div>
      </div>

      {/* Actions Bar */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            placeholder="Search users..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex gap-2">
          <select
            value={selectedRole}
            onChange={(e) => setSelectedRole(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Roles</option>
            <option value="Admin">Admin</option>
            <option value="Developer">Developer</option>
            <option value="Viewer">Viewer</option>
          </select>
          <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
            <Filter className="w-4 h-4" />
            Filter
          </button>
          <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
            <Download className="w-4 h-4" />
            Export
          </button>
          <button 
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            <UserPlus className="w-4 h-4" />
            Add User
          </button>
        </div>
      </div>

      {/* Users Table */}
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="text-left py-3 px-4 font-medium text-gray-700">User</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Role</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Status</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Last Login</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {filteredUsers.map((user) => (
              <tr key={user.id} className="hover:bg-gray-50">
                <td className="py-3 px-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center text-sm font-medium">
                      {user.avatar}
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{user.name}</p>
                      <p className="text-sm text-gray-500">{user.email}</p>
                    </div>
                  </div>
                </td>
                <td className="py-3 px-4">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRoleBadgeColor(user.role)}`}>
                    {user.role}
                  </span>
                </td>
                <td className="py-3 px-4">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadgeColor(user.status)}`}>
                    {user.status}
                  </span>
                </td>
                <td className="py-3 px-4 text-sm text-gray-600">
                  {user.lastLogin}
                </td>
                <td className="py-3 px-4">
                  <div className="relative">
                    <button
                      onClick={() => setShowDropdown(showDropdown === user.id ? null : user.id)}
                      className="p-1 hover:bg-gray-100 rounded"
                    >
                      <MoreVertical className="w-5 h-5 text-gray-500" />
                    </button>
                    {showDropdown === user.id && (
                      <div className="absolute right-0 mt-1 w-48 bg-white rounded-lg shadow-lg border border-gray-200 z-10">
                        <button className="w-full text-left px-4 py-2 hover:bg-gray-50 text-sm">Edit User</button>
                        <button className="w-full text-left px-4 py-2 hover:bg-gray-50 text-sm">Change Role</button>
                        <button className="w-full text-left px-4 py-2 hover:bg-gray-50 text-sm">Reset Password</button>
                        <hr className="my-1" />
                        <button className="w-full text-left px-4 py-2 hover:bg-gray-50 text-sm text-red-600">
                          {user.status === 'Active' ? 'Deactivate' : 'Activate'}
                        </button>
                      </div>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Add User Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Add New User</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input
                  type="email"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                  <option>Viewer</option>
                  <option>Developer</option>
                  <option>Admin</option>
                </select>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowAddModal(false)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                Add User
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UserManagementDashboard;