import React, { useState, useEffect } from 'react';

const Dashboard = () => {
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

  const cardStyle = {
    backgroundColor: 'white',
    padding: '20px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    textAlign: 'center'
  };

  const containerStyle = {
    padding: '20px',
    backgroundColor: '#f5f5f5',
    minHeight: '100vh',
    fontFamily: 'Arial, sans-serif'
  };

  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '20px',
    marginBottom: '30px'
  };

  if (loading) {
    return (
      <div style={containerStyle}>
        <h1>Dashboard</h1>
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <h1 style={{ marginBottom: '30px', color: '#333' }}>
        ML Dashboard
      </h1>
      
      <div style={gridStyle}>
        <div style={cardStyle}>
          <h3 style={{ color: '#666', marginBottom: '10px' }}>Total Models</h3>
          <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#007bff' }}>
            {stats.totalModels}
          </div>
        </div>

        <div style={cardStyle}>
          <h3 style={{ color: '#666', marginBottom: '10px' }}>Active Training</h3>
          <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#ffc107' }}>
            {stats.activeTraining}
          </div>
        </div>

        <div style={cardStyle}>
          <h3 style={{ color: '#666', marginBottom: '10px' }}>Completed Jobs</h3>
          <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#28a745' }}>
            {stats.completedJobs}
          </div>
        </div>

        <div style={cardStyle}>
          <h3 style={{ color: '#666', marginBottom: '10px' }}>Best Accuracy</h3>
          <div style={{ fontSize: '2em', fontWeight: 'bold', color: '#dc3545' }}>
            {stats.accuracy}%
          </div>
        </div>
      </div>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '15px', color: '#333' }}>Recent Activity</h3>
        <div style={{ textAlign: 'left' }}>
          <div style={{ padding: '10px', borderBottom: '1px solid #eee' }}>
            ✅ Model training completed - Random Forest (94.5% accuracy)
          </div>
          <div style={{ padding: '10px', borderBottom: '1px solid #eee' }}>
            🔄 Neural Network training in progress...
          </div>
          <div style={{ padding: '10px', borderBottom: '1px solid #eee' }}>
            📊 New dataset uploaded - creditcard.csv
          </div>
          <div style={{ padding: '10px' }}>
            ⚙️ XGBoost model started training
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;