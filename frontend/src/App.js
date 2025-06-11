import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import ModelConfig from './pages/ModelConfig';
// import DataUpload from './pages/DataUpload';
// import ModelTraining from './pages/ModelTraining';
// import Results from './pages/Results';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/config" element={<ModelConfig />} />
          {/* <Route path="/upload" element={<DataUpload />} /> */}
          {/* <Route path="/train" element={<ModelTraining />} /> */}
          {/* <Route path="/results" element={<Results />} /> */}
        </Routes>
      </div>
    </Router>
  );
}

export default App;