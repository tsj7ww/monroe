import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Home from './pages/Home';
import Train from './pages/Train';
import Serve from './pages/Serve';
import Data from './pages/Data';
import Settings from './pages/Settings';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/train" element={<Train />} />
            <Route path="/serve" element={<Serve />} />
            <Route path="/data" element={<Data />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

const headerLinkStyle = {
  className: 'text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors',
};

function Header() {
  const location = useLocation();
  
  // Function to determine if a link is active
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  // Function to get the appropriate className for each link
  const getLinkClassName = (path) => {
    const baseStyle = headerLinkStyle.className;
    const activeStyle = "bg-blue-100 text-blue-700 border-b-2 border-blue-500";
    
    return isActive(path) ? `${baseStyle} ${activeStyle}` : baseStyle;
  };

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
            <Link to="/" className={getLinkClassName('/')}>
              Home
            </Link>
            <Link to="/train" className={getLinkClassName('/train')}>
              Train
            </Link>
            <Link to="/serve" className={getLinkClassName('/serve')}>
              Serve
            </Link>
            <Link to="/data" className={getLinkClassName('/data')}>
              Data
            </Link>
            <Link to="/settings" className={getLinkClassName('/settings')}>
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
    <footer className="bg-gray-100 border-t border-gray-200 py-4 px-6">
      <div className="max-w-8xl mx-auto text-center">
        <p className="text-gray-600 text-sm">&copy; 2025 AutoML Platform. All rights reserved.</p>
      </div>
    </footer>
  );
}

export default App;