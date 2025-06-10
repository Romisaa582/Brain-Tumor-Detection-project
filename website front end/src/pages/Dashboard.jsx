import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';

const Dashboard = () => {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalPatients: 0,
    todayAppointments: 0,
    pendingResults: 0,
    completedResults: 0
  });
  const [recentPatients, setRecentPatients] = useState([]);
  const [upcomingAppointments, setUpcomingAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (user && user.id && user.role === 'Doctor') {
      fetchDashboardData();
    } else {
      setError('Please log in as a doctor to view the dashboard');
      setLoading(false);
    }
  }, [user]);

  const safeJsonParse = async (response, endpoint) => {
    try {
      const text = await response.text(); // Get raw response text
      if (!text) {
        console.error(`Empty response from ${endpoint}`);
        throw new Error('Server returned empty response');
      }
      
      try {
        return JSON.parse(text);
      } catch (e) {
        console.error(`Invalid JSON from ${endpoint}:`, text);
        throw new Error('Server returned invalid data');
      }
    } catch (err) {
      console.error(`Error parsing response from ${endpoint}:`, err);
      throw err;
    }
  };

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      const token = localStorage.getItem('token');
      
      if (!token) {
        throw new Error('Authentication required');
      }

      const headers = {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      };

      // Log the request details
      console.log('Fetching dashboard data for user:', user.id);

      // Fetch all data in parallel for better performance
      const [statsResponse, patientsResponse, appointmentsResponse] = await Promise.all([
        fetch(`https://tumortraker12.runasp.net/api/Dashboard/doctor/${user.id}/stats`, { headers }),
        fetch(`https://tumortraker12.runasp.net/api/Dashboard/doctor/${user.id}/recent-patients`, { headers }),
        fetch(`https://tumortraker12.runasp.net/api/Dashboard/doctor/${user.id}/upcoming-appointments`, { headers })
      ]);

      // Log response status codes
      console.log('Response status codes:', {
        stats: statsResponse.status,
        patients: patientsResponse.status,
        appointments: appointmentsResponse.status
      });

      // Check if any request failed
      if (!statsResponse.ok) {
        const errorData = await safeJsonParse(statsResponse, 'stats');
        throw new Error(errorData?.message || `Stats request failed with status ${statsResponse.status}`);
      }

      if (!patientsResponse.ok) {
        const errorData = await safeJsonParse(patientsResponse, 'patients');
        throw new Error(errorData?.message || `Patients request failed with status ${patientsResponse.status}`);
      }

      if (!appointmentsResponse.ok) {
        const errorData = await safeJsonParse(appointmentsResponse, 'appointments');
        throw new Error(errorData?.message || `Appointments request failed with status ${appointmentsResponse.status}`);
      }

      // Parse all responses with safe JSON parsing
      const [statsData, patientsData, appointmentsData] = await Promise.all([
        safeJsonParse(statsResponse, 'stats'),
        safeJsonParse(patientsResponse, 'patients'),
        safeJsonParse(appointmentsResponse, 'appointments')
      ]);

      // Validate the data structure
      if (!statsData || typeof statsData !== 'object') {
        throw new Error('Invalid stats data received');
      }

      // Set the data with validation
      setStats({
        totalPatients: statsData.totalPatients || 0,
        todayAppointments: statsData.todayAppointments || 0,
        pendingResults: statsData.pendingResults || 0,
        completedResults: statsData.completedResults || 0
      });
      setRecentPatients(Array.isArray(patientsData) ? patientsData : []);
      setUpcomingAppointments(Array.isArray(appointmentsData) ? appointmentsData : []);

    } catch (err) {
      console.error('Dashboard Error:', err);
      setError(err.message || 'Failed to fetch dashboard data');
      // Set default values on error
      setStats({
        totalPatients: 0,
        todayAppointments: 0,
        pendingResults: 0,
        completedResults: 0
      });
      setRecentPatients([]);
      setUpcomingAppointments([]);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="bg-red-50 border border-red-200 text-red-600 rounded-lg p-4">
          <p>Error: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-lg overflow-hidden mb-8"
        >
          <div className="bg-gradient-to-r from-indigo-600 to-blue-500 px-8 py-12">
            <div className="flex items-center">
              <img
                src={user.photoURL || 'https://via.placeholder.com/160?text=Doctor'}
                alt={user.firstName}
                className="h-24 w-24 rounded-full border-4 border-white shadow-lg object-cover"
              />
              <div className="ml-6 text-white">
                <h1 className="text-3xl font-bold">
                  Welcome back, Dr. {user.firstName} {user.lastName}
                </h1>
                <p className="mt-2 text-indigo-100">Here's what's happening with your patients today.</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Statistics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white rounded-xl shadow-sm p-6"
          >
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-indigo-50">
                <svg className="h-8 w-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Patients</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.totalPatients}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white rounded-xl shadow-sm p-6"
          >
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-green-50">
                <svg className="h-8 w-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Today's Appointments</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.todayAppointments}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-xl shadow-sm p-6"
          >
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-yellow-50">
                <svg className="h-8 w-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Pending Results</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.pendingResults}</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-xl shadow-sm p-6"
          >
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-blue-50">
                <svg className="h-8 w-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Completed Results</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.completedResults}</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Recent Patients and Upcoming Appointments */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Patients */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-white rounded-xl shadow-sm"
          >
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-medium text-gray-900">Recent Patients</h2>
                <Link
                  to="/my-patients"
                  className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
                >
                  View all
                </Link>
              </div>
              <div className="space-y-4">
                {recentPatients.map((patient) => (
                  <div key={patient.id} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <img
                        src={patient.photoURL || 'https://via.placeholder.com/40?text=P'}
                        alt={patient.firstName}
                        className="h-10 w-10 rounded-full object-cover"
                      />
                      <div className="ml-4">
                        <p className="text-sm font-medium text-gray-900">
                          {patient.firstName} {patient.lastName}
                        </p>
                        <p className="text-sm text-gray-500">{patient.lastVisit}</p>
                      </div>
                    </div>
                    <Link
                      to={`/patient-profile/${patient.id}`}
                      className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
                    >
                      View Profile
                    </Link>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Upcoming Appointments */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className="bg-white rounded-xl shadow-sm"
          >
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-medium text-gray-900">Upcoming Appointments</h2>
                <Link
                  to="/my-appointments"
                  className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
                >
                  View all
                </Link>
              </div>
              <div className="space-y-4">
                {upcomingAppointments.map((appointment) => (
                  <div key={appointment.id} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div className="p-2 rounded-full bg-indigo-50">
                        <svg className="h-6 w-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                      </div>
                      <div className="ml-4">
                        <p className="text-sm font-medium text-gray-900">
                          {appointment.patientName}
                        </p>
                        <p className="text-sm text-gray-500">
                          {appointment.date} at {appointment.time}
                        </p>
                      </div>
                    </div>
                    <span className={`inline-flex rounded-full px-2 text-xs font-semibold leading-5 ${
                      appointment.status === 'Confirmed'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {appointment.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 