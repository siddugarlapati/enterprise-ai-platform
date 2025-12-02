import React, { useEffect, useState } from 'react';
import { Grid, Card, CardContent, Typography, Box, CircularProgress } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import axios from 'axios';

interface Stats {
  users: { total: number; active: number };
  transactions: { total: number; last_24h: number };
  models: { total: number; deployed: number };
  engine: { total_requests: number; average_latency_ms: number };
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await axios.get('/api/v1/admin/stats', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setStats(response.data);
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  if (loading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}><CircularProgress /></Box>;
  }

  const statCards = [
    { title: 'Total Users', value: stats?.users.total || 0, subtitle: `${stats?.users.active || 0} active` },
    { title: 'Transactions', value: stats?.transactions.total || 0, subtitle: `${stats?.transactions.last_24h || 0} last 24h` },
    { title: 'Models', value: stats?.models.total || 0, subtitle: `${stats?.models.deployed || 0} deployed` },
    { title: 'Avg Latency', value: `${stats?.engine.average_latency_ms || 0}ms`, subtitle: `${stats?.engine.total_requests || 0} requests` },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Dashboard</Typography>
      <Grid container spacing={3}>
        {statCards.map((card, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>{card.title}</Typography>
                <Typography variant="h4">{card.value}</Typography>
                <Typography variant="body2" color="text.secondary">{card.subtitle}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Request Volume</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={[{ name: 'Now', requests: stats?.engine.total_requests || 0 }]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="requests" stroke="#1976d2" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
