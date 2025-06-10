import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';
import { FaUserInjured, FaClock, FaCalendarAlt } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';

const MyPatients = () => {
  const { user } = useAuth();
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchDoctorAppointments = async () => {
      try {
        setLoading(true);
        setError(null);

        const storedUser = localStorage.getItem('user');
        const userData = JSON.parse(storedUser);
        const token = localStorage.getItem('token');

        if (!userData?.id || !token) {
          throw new Error('Please log in to view patients');
        }

        const response = await fetch(`https://tumortraker12.runasp.net/api/Appointment/GetDoctorAppointment?DoctorId=${userData.id}`, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          const errorText = await response.text();
          console.error('API Error:', {
            status: response.status,
            statusText: response.statusText,
            error: errorText
          });
          throw new Error('Failed to fetch patients');
        }

        const data = await response.json();
        setAppointments(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error('Error:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDoctorAppointments();
  }, []);

  const handleViewDetails = (patientId) => {
    navigate(`/patient-profile/${patientId}`);
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
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-lg overflow-hidden mb-8"
        >
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-8 py-12">
            <h1 className="text-4xl font-bold text-white mb-2">My Patients</h1>
            <p className="text-blue-100 text-lg">Manage your patient appointments and schedules</p>
            <div className="mt-6 flex items-center space-x-4">
              <div className="bg-white/20 rounded-lg px-4 py-2 text-white">
                <span className="font-semibold text-2xl">{appointments.length}</span>
                <span className="ml-2">Total Appointments</span>
              </div>
            </div>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {appointments.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="col-span-full bg-white rounded-xl shadow-sm p-8 text-center"
            >
              <div className="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                <FaUserInjured className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Appointments Found</h3>
              <p className="text-gray-500">You don't have any patient appointments scheduled at the moment.</p>
            </motion.div>
          ) : (
            appointments.map((appointment, index) => (
              <motion.div
                key={appointment.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300"
              >
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center">
                      <FaUserInjured className="w-6 h-6 text-indigo-600" />
                    </div>
                    <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                      Active
                    </span>
                  </div>
                  
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">
                    {appointment.patientName}
                  </h3>
                  
                  <div className="space-y-3">
                    <div className="flex items-center text-gray-600">
                      <FaCalendarAlt className="w-5 h-5 text-indigo-500 mr-3" />
                      <span>{appointment.dayOfWeek}</span>
                    </div>
                    
                    <div className="flex items-center text-gray-600">
                      <FaClock className="w-5 h-5 text-indigo-500 mr-3" />
                      <span>{appointment.time}</span>
                    </div>
                  </div>

                  <div className="mt-6 pt-4 border-t border-gray-100">
                    <button 
                      onClick={() => handleViewDetails(appointment.patientId)}
                      className="w-full bg-indigo-50 hover:bg-indigo-100 text-indigo-600 font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
                    >
                      <span>View Patient Profile</span>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                      </svg>
                    </button>
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default MyPatients; 