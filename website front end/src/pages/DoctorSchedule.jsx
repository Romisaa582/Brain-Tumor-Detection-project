import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';

const DoctorSchedule = () => {
  const { user } = useAuth();
  const [selectedDay, setSelectedDay] = useState(null);
  const [selectedTimeSlots, setSelectedTimeSlots] = useState({});
  const [existingWorkTimes, setExistingWorkTimes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState('');

  const days = [
    { id: 1, name: 'Monday', shortName: 'MON' },
    { id: 2, name: 'Tuesday', shortName: 'TUE' },
    { id: 3, name: 'Wednesday', shortName: 'WED' },
    { id: 4, name: 'Thursday', shortName: 'THU' },
    { id: 5, name: 'Friday', shortName: 'FRI' },
    { id: 6, name: 'Saturday', shortName: 'SAT' },
    { id: 7, name: 'Sunday', shortName: 'SUN' }
  ];

  const timeSlots = [
    '08:00', '08:30', '09:00', '09:30', '10:00', '10:30',
    '11:00', '11:30', '12:00', '12:30', '13:00', '13:30',
    '14:00', '14:30', '15:00', '15:30', '16:00', '16:30',
    '17:00', '17:30', '18:00', '18:30', '19:00', '19:30'
  ];

  const getDayName = (dayId) => {
    const day = days.find(d => d.id === dayId);
    return day ? day.name : 'Sunday';
  };

  useEffect(() => {
    fetchDoctorWorkTimes();
  }, []);

  const fetchDoctorWorkTimes = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      if (!token) {
        throw new Error('Authentication required');
      }

      const response = await fetch(`https://tumortraker12.runasp.net/api/DoctorWorkTime?DoctorId=${user.id}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch work times (Status: ${response.status})`);
      }

      const data = await response.json();
      setExistingWorkTimes(data);

      const formattedSlots = {};
      data.forEach(workTime => {
        // Find the day ID from the day name
        const day = days.find(d => d.name === workTime.day);
        const dayId = day ? day.id : 7; // Default to Sunday if not found
        
        if (!formattedSlots[dayId]) {
          formattedSlots[dayId] = [];
        }

        // Only add the start time if it's a valid time
        if (workTime.startTime && workTime.startTime !== "00:00") {
          // Extract just the HH:mm part
          const timeStr = workTime.startTime.substring(0, 5);
          if (timeSlots.includes(timeStr)) {
            formattedSlots[dayId].push(timeStr);
          }
        }
      });

      setSelectedTimeSlots(formattedSlots);
      setSelectedDay(days[0].id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDaySelect = (dayId) => {
    setSelectedDay(dayId);
  };

  const handleTimeSlotToggle = (time) => {
    setSelectedTimeSlots(prev => {
      const daySlots = [...(prev[selectedDay] || [])];
      const index = daySlots.indexOf(time);
      
      if (index === -1) {
        daySlots.push(time);
        daySlots.sort();
      } else {
        daySlots.splice(index, 1);
      }

      return {
        ...prev,
        [selectedDay]: daySlots
      };
    });
  };

  const handleDeleteSlot = async (dayId, time) => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication required');

      // Find the appointment to delete
      const dayName = getDayName(parseInt(dayId));
      const appointment = existingWorkTimes.find(wt => 
        wt.day === dayName && 
        wt.startTime.startsWith(time)
      );

      if (!appointment) {
        throw new Error('Appointment not found');
      }

      const deleteResponse = await fetch(`https://tumortraker12.runasp.net/api/DoctorWorkTime`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ id: appointment.id })
      });

      if (!deleteResponse.ok) {
        const errorText = await deleteResponse.text();
        throw new Error(`Failed to delete appointment: ${errorText}`);
      }

      // Update local state
      setSelectedTimeSlots(prev => ({
        ...prev,
        [dayId]: prev[dayId].filter(t => t !== time)
      }));

      setSuccessMessage('Appointment deleted successfully');
      setTimeout(() => setSuccessMessage(''), 3000);
      
      // Refresh the work times
      await fetchDoctorWorkTimes();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSchedule = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccessMessage('');
      
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication required');

      // Delete existing appointments first
      try {
        for (const existingTime of existingWorkTimes) {
          if (existingTime.id) {
            const deleteResponse = await fetch(`https://tumortraker12.runasp.net/api/DoctorWorkTime`, {
              method: 'DELETE',
              headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({ id: existingTime.id })
            });

            if (!deleteResponse.ok) {
              const errorText = await deleteResponse.text();
              console.warn(`Could not delete appointment ${existingTime.id}: ${errorText}`);
              if (deleteResponse.status === 404) {
                // If the appointment doesn't exist, we can continue
                continue;
              }
              // For other errors, we should stop
              throw new Error(`Failed to delete appointment: ${errorText}`);
            }
          }
        }
      } catch (deleteErr) {
        setError(deleteErr.message);
        return; // Stop if we can't delete existing appointments
      }

      const workTimes = Object.entries(selectedTimeSlots).flatMap(([dayId, times]) => {
        if (!times || times.length === 0) return [];

        times.sort();

        return times.map(time => {
          const [startHour, startMinute] = time.split(':').map(Number);
          let endHour = startHour;
          let endMinute = startMinute + 30;

          if (endMinute >= 60) {
            endHour += 1;
            endMinute -= 60;
          }

          const dayName = getDayName(parseInt(dayId));

          const formatTime = (hour, minute) => `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;

          return {
            id: 0,
            doctorId: user.id,
            doctorName: user.firstName + ' ' + user.lastName,
            day: dayName,
            startTime: formatTime(startHour, startMinute),
            endTime: formatTime(endHour, endMinute)
          };
        });
      });

      // Save new appointments
      for (const workTime of workTimes) {
        const response = await fetch(`https://tumortraker12.runasp.net/api/DoctorWorkTime`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(workTime)
        });

        if (!response.ok) {
          const msg = await response.text();
          throw new Error(`Failed to save ${workTime.day} ${workTime.startTime}: ${msg}`);
        }
      }

      setSuccessMessage('Work schedule saved successfully!');
      setTimeout(() => setSuccessMessage(''), 3000);
      await fetchDoctorWorkTimes();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="min-h-screen flex items-center justify-center">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-600"></div>
    </div>;
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-white rounded-2xl shadow-lg overflow-hidden mb-8">
          <div className="bg-gradient-to-r from-indigo-600 to-blue-500 px-8 py-12">
            <div className="flex items-center">
              <img 
                src={user.photoURL || 'https://via.placeholder.com/160?text=Doctor'} 
                alt={user.firstName || 'Doctor'} 
                className="h-24 w-24 rounded-full border-4 border-white shadow-lg object-cover" 
              />
              <div className="ml-6 text-white">
                <h1 className="text-3xl font-bold">Work Schedule</h1>
                <p className="mt-2 text-indigo-100">Set your available time slots for appointments.</p>
              </div>
            </div>
          </div>
        </motion.div>

        {error && <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg"><p className="text-sm text-red-600">{error}</p></div>}
        {successMessage && <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg"><p className="text-sm text-green-600">{successMessage}</p></div>}

        <div className="bg-white rounded-xl shadow-sm p-6">
          <div className="mb-8">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Select Day</h2>
            <div className="grid grid-cols-7 gap-2">
              {days.map((day) => (
                <motion.button
                  key={day.id}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleDaySelect(day.id)}
                  className={`p-4 rounded-lg text-center transition-colors ${
                    selectedDay === day.id
                      ? 'bg-indigo-600 text-white'
                      : selectedTimeSlots[day.id]?.length > 0
                      ? 'bg-green-100 text-gray-700 hover:bg-green-200'
                      : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <p className="text-sm font-medium">{day.shortName}</p>
                  <p className="text-xs mt-1">{selectedTimeSlots[day.id]?.length || 0} slots</p>
                </motion.button>
              ))}
            </div>
          </div>

          <div>
            <h2 className="text-lg font-medium text-gray-900 mb-4">Select Time Slots</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
              {timeSlots.map((time) => (
                <div key={time} className="relative">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleTimeSlotToggle(time)}
                    className={`w-full p-3 rounded-lg text-center transition-colors ${
                      selectedTimeSlots[selectedDay]?.includes(time)
                        ? 'bg-green-600 text-white'
                        : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    {time}
                  </motion.button>
                  {selectedTimeSlots[selectedDay]?.includes(time) && (
                    <button
                      onClick={() => handleDeleteSlot(selectedDay, time)}
                      className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs hover:bg-red-600 focus:outline-none"
                      title="Delete this time slot"
                    >
                      ×
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="mt-8 flex justify-end">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleSaveSchedule}
              disabled={loading}
              className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Saving...' : 'Save Schedule'}
            </motion.button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DoctorSchedule;
