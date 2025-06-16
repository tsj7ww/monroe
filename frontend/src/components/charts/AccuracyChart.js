import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const AccuracyChart = ({ modelData, width = 500, height = 350 }) => {
  // Generate mock data for demonstration - replace with actual model data
  const generateMockData = (modelType, accuracy) => {
    const dataPoints = [];
    const numPoints = 100;
    const errorVariance = modelType === 'Classification' ? 0.1 : (1 - accuracy) * 2;
    
    for (let i = 0; i < numPoints; i++) {
      let predicted, actual;
      
      if (modelType === 'Classification') {
        // For classification: binary values (0 or 1) with some noise
        const trueValue = Math.random() > 0.5 ? 1 : 0;
        predicted = Math.max(0, Math.min(1, trueValue + (Math.random() - 0.5) * errorVariance));
        actual = trueValue;
      } else {
        // For regression: continuous values
        const baseValue = Math.random() * 100;
        predicted = baseValue + (Math.random() - 0.5) * errorVariance * 20;
        actual = baseValue + (Math.random() - 0.5) * errorVariance * 10;
      }
      
      dataPoints.push({
        predicted: Number(predicted.toFixed(2)),
        actual: Number(actual.toFixed(2)),
        id: i
      });
    }
    
    return dataPoints;
  };

  const data = modelData?.predictions || generateMockData(modelData?.type || 'Regression', modelData?.accuracy || 0.85);
  
  // Calculate R² for display
  const calculateRSquared = (data) => {
    const n = data.length;
    const actualMean = data.reduce((sum, point) => sum + point.actual, 0) / n;
    
    const totalSumSquares = data.reduce((sum, point) => sum + Math.pow(point.actual - actualMean, 2), 0);
    const residualSumSquares = data.reduce((sum, point) => sum + Math.pow(point.actual - point.predicted, 2), 0);
    
    return Math.max(0, 1 - (residualSumSquares / totalSumSquares));
  };

  const rSquared = calculateRSquared(data);
  
  // Determine axis ranges
  const allValues = data.flatMap(d => [d.predicted, d.actual]);
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const padding = (maxValue - minValue) * 0.1;
  const axisMin = Math.round(minValue - padding, 2);
  const axisMax = Math.round(maxValue + padding, 2);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="text-sm font-medium text-gray-900 mb-1">Data Point</p>
          <p className="text-xs text-gray-600">Predicted: <span className="font-medium">{data.predicted}</span></p>
          <p className="text-xs text-gray-600">Actual: <span className="font-medium">{data.actual}</span></p>
          <p className="text-xs text-gray-600">Error: <span className="font-medium">{Math.abs(data.predicted - data.actual).toFixed(3)}</span></p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <div className="mb-4 flex justify-between items-center">
        <div>
          <h4 className="text-lg font-semibold text-gray-900">Prediction Accuracy</h4>
          <p className="text-sm text-gray-600">Predicted vs Actual Values</p>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-600">R² Score</div>
          <div className="text-2xl font-bold text-blue-600">{rSquared.toFixed(3)}</div>
        </div>
      </div>
      
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        <ResponsiveContainer width="100%" height={height}>
          <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis 
              type="number" 
              dataKey="predicted" 
              domain={[axisMin, axisMax]}
              tick={{ fontSize: 12 }}
              label={{ value: 'Predicted Values', position: 'insideBottom', offset: -10, style: { textAnchor: 'middle', fontSize: '12px', fill: '#666' } }}
            />
            <YAxis 
              type="number" 
              dataKey="actual" 
              domain={[axisMin, axisMax]}
              tick={{ fontSize: 12 }}
              label={{ value: 'Actual Values', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fontSize: '12px', fill: '#666' } }}
            />
            <Tooltip content={<CustomTooltip />} />
            
            {/* Perfect prediction line (y = x) */}
            <ReferenceLine 
              segment={[{x: axisMin, y: axisMin}, {x: axisMax, y: axisMax}]} 
              stroke="#ef4444" 
              strokeWidth={2}
              strokeDasharray="5 5"
            />
            
            <Scatter 
              data={data} 
              fill="#3b82f6" 
              fillOpacity={0.6}
              stroke="#1d4ed8"
              strokeWidth={1}
              r={3}
            />
          </ScatterChart>
        </ResponsiveContainer>
        
        <div className="mt-4 flex items-center justify-between text-xs text-gray-600">
          <div className="flex items-center">
            <div className="w-3 h-0.5 bg-red-500 mr-2"></div>
            <span>Perfect Prediction Line</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 rounded-full mr-2 opacity-60"></div>
            <span>Prediction Points ({data.length} samples)</span>
          </div>
        </div>
      </div>
      
      <div className="mt-4 bg-gray-50 p-4 rounded-lg">
        <h5 className="font-medium text-gray-900 mb-2">Chart Interpretation</h5>
        <div className="text-sm text-gray-600 space-y-1">
          <p>• Points closer to the red line indicate more accurate predictions</p>
          <p>• R² score of 1.0 represents perfect predictions</p>
          <p>• Higher R² values indicate better model performance</p>
        </div>
      </div>
    </div>
  );
};

export default AccuracyChart;