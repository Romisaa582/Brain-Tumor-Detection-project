import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const BookAppointment = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState('');
  const [doctors, setDoctors] = useState([]);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [selectedDay, setSelectedDay] = useState(null);
  const [selectedTime, setSelectedTime] = useState(null);
  const [availableSlots, setAvailableSlots] = useState({});

  const days = [
    { id: 0, name: 'Sunday', shortName: 'SUN' },
    { id: 1, name: 'Monday', shortName: 'MON' },
    { id: 2, name: 'Tuesday', shortName: 'TUE' },
    { id: 3, name: 'Wednesday', shortName: 'WED' },
    { id: 4, name: 'Thursday', shortName: 'THU' },
    { id: 5, name: 'Friday', shortName: 'FRI' },
    { id: 6, name: 'Saturday', shortName: 'SAT' }
  ];

  useEffect(() => {
    fetchDoctors();
  }, []);

  useEffect(() => {
    if (selectedDoctor && selectedDay !== null) {
      fetchAvailableSlots();
    }
  }, [selectedDoctor, selectedDay]);

  const fetchDoctors = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      if (!token) {
        throw new Error('Authentication required');
      }

      const response = await fetch('https://tumortraker12.runasp.net/api/Doctor', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch doctors');
      }

      const data = await response.json();
      setDoctors(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchAvailableSlots = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      if (!token) {
        throw new Error('يرجى تسجيل الدخول أولاً');
      }

      const response = await fetch(`https://tumortraker12.runasp.net/api/DoctorWorkTime?DoctorId=${selectedDoctor.id}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error('فشل في جلب المواعيد المتاحة');
      }

      const data = await response.json();
      
      // تصفية المواعيد لليوم المحدد
      const daySlots = data.filter(slot => slot.day === days[selectedDay].name)
        .map(slot => ({
          ...slot,
          time: slot.startTime // استخدام startTime فقط
        }));
      setAvailableSlots(daySlots);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBookAppointment = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication required');

      // Calculate end time (assuming 30 min appointments)
      const [startHour, startMinute] = selectedTime.split(':').map(Number);
      let endHour = startHour;
      let endMinute = startMinute + 30;
      
      if (endMinute >= 60) {
        endHour += 1;
        endMinute -= 60;
      }

      const startTime = `${String(startHour).padStart(2, '0')}:${String(startMinute).padStart(2, '0')}`;
      const endTime = `${String(endHour).padStart(2, '0')}:${String(endMinute).padStart(2, '0')}`;

      const appointment = {
        id: 0,
        doctorId: selectedDoctor.id,
        day: days[selectedDay].name,
        startTime: startTime,
        endTime: endTime
      };

      const response = await fetch('https://tumortraker12.runasp.net/api/Appointment', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(appointment)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.message || 'Failed to book appointment');
      }

      const data = await response.json();
      setSuccessMessage('Appointment booked successfully!');
      setTimeout(() => {
        setSuccessMessage('');
        navigate('/appointments'); // Redirect to appointments page
      }, 2000);
    } catch (err) {
      setError(err.message);
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

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-lg overflow-hidden mb-8"
        >
          <div className="bg-gradient-to-r from-indigo-600 to-blue-500 px-8 py-12">
            <h1 className="text-3xl font-bold text-white">Book an Appointment</h1>
            <p className="mt-2 text-indigo-100">Select a doctor and available time slot.</p>
          </div>
        </motion.div>

        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {successMessage && (
          <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-sm text-green-600">{successMessage}</p>
          </div>
        )}

        <div className="bg-white rounded-xl shadow-sm p-6">
          {/* Doctor Selection */}
          <div className="mb-8">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Select Doctor</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {doctors.map((doctor) => (
                <motion.button
                  key={doctor.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setSelectedDoctor(doctor)}
                  className={`p-4 rounded-lg border transition-colors ${
                    selectedDoctor?.id === doctor.id
                      ? 'border-indigo-600 bg-indigo-50'
                      : 'border-gray-200 hover:border-indigo-300'
                  }`}
                >
                  <div className="flex items-center">
                    <img
                      src={doctor.photoURL || 'https://via.placeholder.com/40?text=Dr'}
                      alt={doctor.name}
                      className="w-10 h-10 rounded-full object-cover"
                    />
                    <div className="ml-3 text-left">
                      <p className="font-medium text-gray-900">Dr. {doctor.firstName} {doctor.lastName}</p>
                      <p className="text-sm text-gray-500">{doctor.specialization || 'Specialist'}</p>
                    </div>
                  </div>
                </motion.button>
              ))}
            </div>
          </div>

          {/* Day Selection */}
          {selectedDoctor && (
            <div className="mb-8">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Select Day</h2>
              <div className="grid grid-cols-7 gap-2">
                {days.map((day) => (
                  <motion.button
                    key={day.id}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => {
                      setSelectedDay(day.id);
                      setSelectedTime(null);
                    }}
                    className={`p-4 rounded-lg text-center transition-colors ${
                      selectedDay === day.id
                        ? 'bg-indigo-600 text-white'
                        : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <p className="text-sm font-medium">{day.shortName}</p>
                  </motion.button>
                ))}
              </div>
            </div>
          )}

          {/* Time Slots */}
          {selectedDay !== null && availableSlots.length > 0 && (
            <div className="mb-8">
              <h2 className="text-lg font-medium text-gray-900 mb-4">اختر الوقت</h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
                {availableSlots.map((slot) => (
                  <motion.button
                    key={slot.time}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setSelectedTime(slot.time)}
                    className={`p-3 rounded-lg text-center transition-colors ${
                      selectedTime === slot.time
                        ? 'bg-green-600 text-white'
                        : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    {slot.time}
                  </motion.button>
                ))}
              </div>
            </div>
          )}

          {/* Book Button */}
          {selectedDoctor && selectedDay !== null && selectedTime && (
            <div className="flex justify-end">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleBookAppointment}
                disabled={loading}
                className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Booking...' : 'Book Appointment'}
              </motion.button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BookAppointment; 