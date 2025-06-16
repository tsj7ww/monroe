import React, { useState } from 'react';

const ModelConfigSearch = ({ 
    onSelect, 
    onCancel, 
    hideSelectButton = false,
    onRowClick,
    selectable = true,
    customActionButton,
    title = "Search Model Configurations"
}) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedConfigId, setSelectedConfigId] = useState(null);
    const [configs] = useState([
        { id: 1, name: 'CNN Image Classifier', type: 'CNN', learningRate: 0.001, accuracy: 92.5, createdAt: '2024-01-15' },
        { id: 2, name: 'LSTM Text Analyzer', type: 'LSTM', learningRate: 0.01, accuracy: 87.3, createdAt: '2024-01-10' },
        { id: 3, name: 'Random Forest Predictor', type: 'Random Forest', learningRate: 0.1, accuracy: 89.1, createdAt: '2024-01-08' },
    ]);

    const filteredConfigs = configs.filter(config =>
        config.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        config.type.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const handleRowClick = (config) => {
        if (selectable) {
            setSelectedConfigId(config.id);
        }
        if (onRowClick) {
            onRowClick(config);
        }
    };

    return (
        <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold">{title}</h3>
                <button onClick={onCancel} className="text-gray-500 hover:text-gray-700">
                    ✕
                </button>
            </div>

            <div className="mb-6">
                <input
                    type="text"
                    placeholder="Search configurations..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
            </div>

            <div className="space-y-4">
                {filteredConfigs.map((config) => (
                    <div 
                        key={config.id} 
                        className={`border rounded-lg p-4 transition-colors cursor-pointer ${
                            selectedConfigId === config.id 
                                ? 'border-blue-500 bg-blue-50' 
                                : 'border-gray-200 hover:bg-gray-50'
                        }`}
                        onClick={() => handleRowClick(config)}
                    >
                        <div className="flex items-center justify-between">
                            <div>
                                <h4 className="font-medium text-gray-900">{config.name}</h4>
                                <p className="text-sm text-gray-600">
                                    Type: {config.type} • Learning Rate: {config.learningRate} • Accuracy: {config.accuracy}%
                                </p>
                                <p className="text-xs text-gray-500">Created: {config.createdAt}</p>
                            </div>
                            {!hideSelectButton && (
                                customActionButton ? (
                                    customActionButton(config)
                                ) : (
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onSelect(config);
                                        }}
                                        className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                                    >
                                        Select
                                    </button>
                                )
                            )}
                        </div>
                    </div>
                ))}
            </div>

            {filteredConfigs.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                    No configurations found matching your search.
                </div>
            )}
        </div>
    );
};

export default ModelConfigSearch;