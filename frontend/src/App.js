import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Data from './pages/Data';
import Train from './pages/Train';
// import Serve from './pages/Serve';
// import Settings from './pages/Settings';

function App() {
  return (
    <Router>
      <div className="App">
      <Header />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/data" element={<Data />} />
          <Route path="/train" element={<Train />} />
          {/* <Route path="/serve" element={<Serve />} /> */}
          {/* <Route path="/settings" element={<Settings />} /> */}
        </Routes>
      <Footer />
      </div>
    </Router>
  );
}

function Header() {
  return (
    <header className="bg-white shadow-md border-b border-gray-200">
      <nav className="max-w-8xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            {/* <h1 className="text-xl font-bold text-gray-900">ML Dashboard</h1> */}
            <Link to="/" className="text-gray-900 hover:text-blue-900 px-3 py-2 rounded-md text-xl font-bold transition-colors">
              AutoML
            </Link>
          </div>
          <div className="flex space-x-8">
            <Link to="/" className="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">
              Home
            </Link>
            <Link to="/data" className="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">
              Data
            </Link>
            <Link to="/train" className="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">
              Train
            </Link>
            <Link to="/serve" className="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">
              Serve
            </Link>
            <Link to="/settings" className="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">
              Settings
            </Link>
          </div>
        </div>
      </nav>
    </header>
  );
}

function Footer() {
  return (
    <footer>
      <p>&copy; 2025 My Site</p>
    </footer>
  );
}

export default App;