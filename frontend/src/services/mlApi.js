import api from './api';

export const mlApi = {
  uploadData: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/api/data/upload', formData);
  },
  
  trainModel: (config) => api.post('/api/models/train', config),
  
  getModels: () => api.get('/api/models'),
  
  getResults: (modelId) => api.get(`/api/results/${modelId}`)
};