import React, { useState, useEffect } from 'react';
import {
  Box, Typography, Card, CardContent, TextField, Button, Grid, Alert,
  Switch, FormControlLabel, Divider, CircularProgress
} from '@mui/material';
import axios from 'axios';

interface UserProfile {
  id: number;
  username: string;
  email: string;
  full_name: string | null;
  role: string;
}

const Settings: React.FC = () => {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [fullName, setFullName] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  const token = localStorage.getItem('token');
  const headers = { Authorization: `Bearer ${token}` };

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const response = await axios.get('/api/v1/auth/me', { headers });
        setProfile(response.data);
        setFullName(response.data.full_name || '');
      } catch (error) {
        console.error('Failed to fetch profile:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setMessage({ type: '', text: '' });
    try {
      await axios.put(`/api/v1/auth/me?full_name=${encodeURIComponent(fullName)}`, {}, { headers });
      setMessage({ type: 'success', text: 'Profile updated successfully' });
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to update profile' });
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}><CircularProgress /></Box>;
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Settings</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Profile</Typography>
              
              {message.text && (
                <Alert severity={message.type as 'success' | 'error'} sx={{ mb: 2 }}>
                  {message.text}
                </Alert>
              )}

              <TextField fullWidth label="Username" value={profile?.username || ''} margin="normal" disabled />
              <TextField fullWidth label="Email" value={profile?.email || ''} margin="normal" disabled />
              <TextField fullWidth label="Role" value={profile?.role || ''} margin="normal" disabled />
              <TextField fullWidth label="Full Name" value={fullName} margin="normal"
                onChange={(e) => setFullName(e.target.value)} />

              <Button variant="contained" onClick={handleSave} disabled={saving} sx={{ mt: 2 }}>
                {saving ? <CircularProgress size={24} /> : 'Save Changes'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Preferences</Typography>
              <FormControlLabel control={<Switch defaultChecked />} label="Email notifications" />
              <FormControlLabel control={<Switch defaultChecked />} label="Dark mode" />
              <FormControlLabel control={<Switch />} label="Two-factor authentication" />
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="h6" gutterBottom>API Keys</Typography>
              <Typography variant="body2" color="text.secondary">
                Manage your API keys for programmatic access to the platform.
              </Typography>
              <Button variant="outlined" sx={{ mt: 2 }}>Generate New API Key</Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;
