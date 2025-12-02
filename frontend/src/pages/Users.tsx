import React, { useEffect, useState } from 'react';
import {
  Box, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Paper, Chip, IconButton, Select, MenuItem, CircularProgress
} from '@mui/material';
import { Block as BlockIcon, CheckCircle as ActiveIcon } from '@mui/icons-material';
import axios from 'axios';

interface User {
  id: number;
  username: string;
  email: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
  created_at: string;
}

const Users: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);

  const token = localStorage.getItem('token');
  const headers = { Authorization: `Bearer ${token}` };

  const fetchUsers = async () => {
    try {
      const response = await axios.get('/api/v1/admin/users', { headers });
      setUsers(response.data);
    } catch (error) {
      console.error('Failed to fetch users:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchUsers(); }, []);

  const handleRoleChange = async (userId: number, role: string) => {
    try {
      await axios.put(`/api/v1/admin/users/${userId}/role?role=${role}`, {}, { headers });
      fetchUsers();
    } catch (error) {
      console.error('Failed to update role:', error);
    }
  };

  const handleToggleStatus = async (userId: number, isActive: boolean) => {
    try {
      await axios.put(`/api/v1/admin/users/${userId}/status?is_active=${!isActive}`, {}, { headers });
      fetchUsers();
    } catch (error) {
      console.error('Failed to toggle status:', error);
    }
  };

  if (loading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}><CircularProgress /></Box>;
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Users</Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Username</TableCell>
              <TableCell>Email</TableCell>
              <TableCell>Full Name</TableCell>
              <TableCell>Role</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {users.map((user) => (
              <TableRow key={user.id}>
                <TableCell>{user.username}</TableCell>
                <TableCell>{user.email}</TableCell>
                <TableCell>{user.full_name || '-'}</TableCell>
                <TableCell>
                  <Select size="small" value={user.role} onChange={(e) => handleRoleChange(user.id, e.target.value)}>
                    <MenuItem value="user">User</MenuItem>
                    <MenuItem value="analyst">Analyst</MenuItem>
                    <MenuItem value="model_manager">Model Manager</MenuItem>
                    <MenuItem value="admin">Admin</MenuItem>
                  </Select>
                </TableCell>
                <TableCell>
                  <Chip label={user.is_active ? 'Active' : 'Inactive'}
                        color={user.is_active ? 'success' : 'default'} size="small" />
                </TableCell>
                <TableCell>
                  <IconButton onClick={() => handleToggleStatus(user.id, user.is_active)}
                              color={user.is_active ? 'error' : 'success'}>
                    {user.is_active ? <BlockIcon /> : <ActiveIcon />}
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default Users;
