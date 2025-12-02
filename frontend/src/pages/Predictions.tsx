import React, { useState } from 'react';
import {
  Box, Typography, Card, CardContent, TextField, Button, Select, MenuItem,
  FormControl, InputLabel, Grid, Alert, CircularProgress, Chip
} from '@mui/material';
import axios from 'axios';

type TaskType = 'sentiment' | 'ner' | 'classify' | 'similarity';

const Predictions: React.FC = () => {
  const [taskType, setTaskType] = useState<TaskType>('sentiment');
  const [text, setText] = useState('');
  const [text2, setText2] = useState('');
  const [labels, setLabels] = useState('finance, sports, technology');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const token = localStorage.getItem('token');

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      let endpoint = `/api/v1/predictions/${taskType}`;
      let payload: any = { text };

      if (taskType === 'similarity') {
        payload = { text1: text, text2: text2 };
      } else if (taskType === 'classify') {
        payload = { text, labels: labels.split(',').map(l => l.trim()) };
      }

      const response = await axios.post(endpoint, payload, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const renderResult = () => {
    if (!result) return null;

    if (taskType === 'sentiment') {
      return (
        <Box>
          <Typography variant="h6">
            Sentiment: <Chip label={result.sentiment} color={result.sentiment === 'positive' ? 'success' : 'error'} />
          </Typography>
          <Typography>Confidence: {(result.confidence * 100).toFixed(1)}%</Typography>
        </Box>
      );
    }

    if (taskType === 'ner') {
      return (
        <Box>
          <Typography variant="h6">Entities Found: {result.entities?.length || 0}</Typography>
          <Box sx={{ mt: 1 }}>
            {result.entities?.map((e: any, i: number) => (
              <Chip key={i} label={`${e.entity} (${e.label})`} sx={{ mr: 1, mb: 1 }} />
            ))}
          </Box>
        </Box>
      );
    }

    if (taskType === 'classify') {
      return (
        <Box>
          <Typography variant="h6">
            Predicted: <Chip label={result.predicted_label} color="primary" />
          </Typography>
          <Typography>Confidence: {(result.confidence * 100).toFixed(1)}%</Typography>
        </Box>
      );
    }

    if (taskType === 'similarity') {
      return (
        <Box>
          <Typography variant="h6">
            Similarity Score: {(result.similarity_score * 100).toFixed(1)}%
          </Typography>
        </Box>
      );
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Predictions</Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <FormControl fullWidth margin="normal">
                <InputLabel>Task Type</InputLabel>
                <Select value={taskType} label="Task Type" onChange={(e) => setTaskType(e.target.value as TaskType)}>
                  <MenuItem value="sentiment">Sentiment Analysis</MenuItem>
                  <MenuItem value="ner">Named Entity Recognition</MenuItem>
                  <MenuItem value="classify">Text Classification</MenuItem>
                  <MenuItem value="similarity">Semantic Similarity</MenuItem>
                </Select>
              </FormControl>

              <TextField fullWidth label="Text" multiline rows={4} margin="normal"
                value={text} onChange={(e) => setText(e.target.value)} />

              {taskType === 'similarity' && (
                <TextField fullWidth label="Second Text" multiline rows={4} margin="normal"
                  value={text2} onChange={(e) => setText2(e.target.value)} />
              )}

              {taskType === 'classify' && (
                <TextField fullWidth label="Labels (comma-separated)" margin="normal"
                  value={labels} onChange={(e) => setLabels(e.target.value)} />
              )}

              <Button variant="contained" fullWidth onClick={handlePredict} disabled={loading || !text} sx={{ mt: 2 }}>
                {loading ? <CircularProgress size={24} /> : 'Analyze'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Results</Typography>
              {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
              {result ? renderResult() : <Typography color="text.secondary">Run a prediction to see results</Typography>}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Predictions;
