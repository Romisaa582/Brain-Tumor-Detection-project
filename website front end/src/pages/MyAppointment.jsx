import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';

const MyAppointment = () => {
  const { user } = useAuth();
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAppointments = async () => {
      try {
        setLoading(true);
        setError(null);

        const storedUser = localStorage.getItem('user');
        const userData = JSON.parse(storedUser);
        const token = localStorage.getItem('token');

        if (!userData?.id || !token) {
          throw new Error('Please log in to view appointments');
        }

        const response = await fetch(`https://tumortraker12.runasp.net/api/Appointment/GetPatientAppointment?PatientId=${userData.id}`, {
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
          throw new Error('Failed to fetch appointments');
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

    fetchAppointments();
  }, []);

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

  const formatTime = (time) => {
    // Handle time in 24-hour format (00:00)
    if (typeof time === 'string') {
      const [hours, minutes] = time.split(':').map(Number);
      const ampm = hours >= 12 ? 'PM' : 'AM';
      const hours12 = hours % 12 || 12;
      return `${hours12}:${minutes.toString().padStart(2, '0')} ${ampm}`;
    }
    // Handle numeric time (e.g., 0)
    if (typeof time === 'number') {
      const hours = Math.floor(time);
      const minutes = Math.round((time % 1) * 60);
      const ampm = hours >= 12 ? 'PM' : 'AM';
      const hours12 = hours % 12 || 12;
      return `${hours12}:${minutes.toString().padStart(2, '0')} ${ampm}`;
    }
    return '00:00'; // Fallback
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-lg overflow-hidden mb-8"
        >
          <div className="bg-gradient-to-r from-indigo-600 to-blue-500 px-8 py-12">
            <h1 className="text-3xl font-bold text-white">My Appointments</h1>
            <p className="mt-2 text-indigo-100">View and manage your appointments</p>
          </div>
        </motion.div>

        <div className="space-y-4">
          {appointments.length === 0 ? (
            <div className="bg-white rounded-xl shadow-sm p-6 text-center">
              <p className="text-gray-500">No appointments found</p>
            </div>
          ) : (
            appointments.map((appointment) => (
              <motion.div
                key={appointment.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white rounded-xl shadow-sm p-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">
                      Dr. {appointment.doctorName}
                    </h3>
                    <div className="mt-2 space-y-1">
                      <p className="text-sm text-gray-500">
                        Day: {appointment.dayOfWeek}
                      </p>
                      <p className="text-sm text-gray-500">
                        Time: {formatTime(appointment.time)}
                      </p>
                      <p className="text-sm text-gray-500">
                        Patient: {appointment.patientName}
                      </p>
                    </div>
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

export default MyAppointment; 