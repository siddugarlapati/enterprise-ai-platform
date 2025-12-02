import React, { useEffect, useState } from 'react';
import {
  Box, Typography, Card, CardContent, Button, Chip, IconButton,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper,
  Dialog, DialogTitle, DialogContent, DialogActions, TextField, CircularProgress
} from '@mui/material';
import { Add as AddIcon, CloudUpload as DeployIcon, Delete as DeleteIcon } from '@mui/icons-material';
import axios from 'axios';

interface Model {
  id: number;
  model_name: string;
  model_type: string;
  description: string;
  accuracy: number | null;
  version: string;
  deployed: boolean;
  created_at: string;
}

const Models: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newModel, setNewModel] = useState({ model_name: '', model_type: 'random_forest', description: '' });

  const token = localStorage.getItem('token');
  const headers = { Authorization: `Bearer ${token}` };

  const fetchModels = async () => {
    try {
      const response = await axios.get('/api/v1/models/', { headers });
      setModels(response.data);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchModels(); }, []);

  const handleCreate = async () => {
    try {
      await axios.post('/api/v1/models/', newModel, { headers });
      setDialogOpen(false);
      setNewModel({ model_name: '', model_type: 'random_forest', description: '' });
      fetchModels();
    } catch (error) {
      console.error('Failed to create model:', error);
    }
  };

  const handleDeploy = async (id: number, deployed: boolean) => {
    try {
      if (deployed) {
        await axios.delete(`/api/v1/models/${id}/deploy`, { headers });
      } else {
        await axios.post(`/api/v1/models/${id}/deploy`, {}, { headers });
      }
      fetchModels();
    } catch (error) {
      console.error('Failed to toggle deployment:', error);
    }
  };

  if (loading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}><CircularProgress /></Box>;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Models</Typography>
        <Button variant="contained" startIcon={<AddIcon />} onClick={() => setDialogOpen(true)}>
          New Model
        </Button>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Accuracy</TableCell>
              <TableCell>Version</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {models.map((model) => (
              <TableRow key={model.id}>
                <TableCell>{model.model_name}</TableCell>
                <TableCell>{model.model_type}</TableCell>
                <TableCell>{model.accuracy ? `${(model.accuracy * 100).toFixed(1)}%` : 'N/A'}</TableCell>
                <TableCell>{model.version}</TableCell>
                <TableCell>
                  <Chip label={model.deployed ? 'Deployed' : 'Not Deployed'} 
                        color={model.deployed ? 'success' : 'default'} size="small" />
                </TableCell>
                <TableCell>
                  <IconButton onClick={() => handleDeploy(model.id, model.deployed)} color="primary">
                    <DeployIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)}>
        <DialogTitle>Create New Model</DialogTitle>
        <DialogContent>
          <TextField fullWidth label="Model Name" margin="normal"
            value={newModel.model_name} onChange={(e) => setNewModel({ ...newModel, model_name: e.target.value })} />
          <TextField fullWidth label="Model Type" margin="normal" select SelectProps={{ native: true }}
            value={newModel.model_type} onChange={(e) => setNewModel({ ...newModel, model_type: e.target.value })}>
            <option value="random_forest">Random Forest</option>
            <option value="gradient_boosting">Gradient Boosting</option>
            <option value="logistic_regression">Logistic Regression</option>
          </TextField>
          <TextField fullWidth label="Description" margin="normal" multiline rows={3}
            value={newModel.description} onChange={(e) => setNewModel({ ...newModel, description: e.target.value })} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreate} variant="contained">Create</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Models;
