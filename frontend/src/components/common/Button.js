import React from 'react';

const Button = ({ children, onClick, disabled, variant = 'primary' }) => {
  const baseClasses = 'px-4 py-2 rounded font-medium';
  const variants = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-600 text-white hover:bg-gray-700'
  };

  return (
    <button
      className={`${baseClasses} ${variants[variant]} ${disabled ? 'opacity-50' : ''}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
};

export default Button;