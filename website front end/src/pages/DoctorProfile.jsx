import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const DoctorProfile = () => {
  const { docId } = useParams();
  const [doctor, setDoctor] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTab, setSelectedTab] = useState('overview');

  useEffect(() => {
    fetchDoctorDetails();
  }, [docId]);

  const fetchDoctorDetails = async () => {
    try {
      setLoading(true);
      setError(null);
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('Authentication required');
      }

      const response = await fetch(`https://tumortraker12.runasp.net/api/Account/GetUserById?userId=${docId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch doctor details');
      }

      const data = await response.json();
      setDoctor(data);
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

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="bg-red-50 border border-red-200 text-red-600 rounded-lg p-4">
          <p>Error: {error}</p>
        </div>
      </div>
    );
  }

  if (!doctor) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-gray-500">Doctor not found</div>
      </div>
    );
  }

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'experience', label: 'Experience' },
    { id: 'reviews', label: 'Reviews' },
    { id: 'schedule', label: 'Schedule' }
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Profile Header */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-lg overflow-hidden"
        >
          <div className="bg-gradient-to-r from-indigo-600 to-blue-500 h-48 sm:h-64"></div>
          <div className="relative px-4 sm:px-6 lg:px-8 pb-8">
            <div className="relative -mt-24 sm:-mt-32 flex justify-between items-end">
              <div className="flex items-end">
                <img
                  src={doctor.photoURL || 'https://via.placeholder.com/160?text=Doctor'}
                  alt={`Dr. ${doctor.firstName} ${doctor.lastName}`}
                  className="h-40 w-40 sm:h-48 sm:w-48 rounded-2xl border-4 border-white shadow-lg object-cover"
                />
                <div className="ml-6 mb-3">
                  <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">
                    Dr. {doctor.firstName} {doctor.lastName}
                  </h1>
                  <p className="text-indigo-600 font-medium mt-1">{doctor.specialization || 'Specialist'}</p>
                  <div className="flex items-center mt-2">
                    <span className="flex items-center text-yellow-400">
                      ★★★★★
                      <span className="text-gray-600 ml-2">4.8 (124 reviews)</span>
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex space-x-4">
                <Link
                  to={`/patient-profile/${doctor.id}`}
                  className="inline-flex items-center px-6 py-3 border border-indigo-600 text-base font-medium rounded-full shadow-sm text-indigo-600 bg-white hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-300"
                >
                  View Profile
                </Link>
                <Link
                  to={`/appointment/${doctor.id}`}
                  className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-full shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-300"
                >
                  Book Appointment
                </Link>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Navigation Tabs */}
        <div className="mt-8 bg-white rounded-xl shadow-sm">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6" aria-label="Tabs">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setSelectedTab(tab.id)}
                  className={`
                    ${selectedTab === tab.id
                      ? 'border-indigo-500 text-indigo-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                    whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm
                  `}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {selectedTab === 'overview' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="space-y-6"
              >
                <div>
                  <h3 className="text-lg font-medium text-gray-900">About</h3>
                  <p className="mt-4 text-gray-500">
                    Dr. {doctor.firstName} {doctor.lastName} is a highly qualified medical professional specializing in {doctor.specialization || 'medicine'}. 
                    With years of experience in the field, they are dedicated to providing exceptional patient care and utilizing the latest medical technologies.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-gray-50 rounded-lg p-6">
                    <h4 className="text-base font-medium text-gray-900 mb-4">Specializations</h4>
                    <ul className="space-y-3">
                      <li className="flex items-center text-gray-600">
                        <svg className="h-5 w-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        {doctor.specialization || 'General Medicine'}
                      </li>
                      <li className="flex items-center text-gray-600">
                        <svg className="h-5 w-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Cancer Detection
                      </li>
                      <li className="flex items-center text-gray-600">
                        <svg className="h-5 w-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Patient Care
                      </li>
                    </ul>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-6">
                    <h4 className="text-base font-medium text-gray-900 mb-4">Contact Information</h4>
                    <ul className="space-y-3">
                      <li className="flex items-center text-gray-600">
                        <svg className="h-5 w-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                        {doctor.email}
                      </li>
                      <li className="flex items-center text-gray-600">
                        <svg className="h-5 w-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                        </svg>
                        Available on appointment
                      </li>
                      <li className="flex items-center text-gray-600">
                        <svg className="h-5 w-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        Medical Center Location
                      </li>
                    </ul>
                  </div>
                </div>
              </motion.div>
            )}

            {selectedTab === 'experience' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="space-y-6"
              >
                <div className="border-l-2 border-indigo-500 pl-4 space-y-8">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">Senior Specialist</h3>
                    <p className="text-sm text-gray-500">2020 - Present</p>
                    <p className="mt-2 text-gray-600">Leading cancer detection and treatment programs.</p>
                  </div>
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">Research Fellow</h3>
                    <p className="text-sm text-gray-500">2018 - 2020</p>
                    <p className="mt-2 text-gray-600">Conducted research on early cancer detection methods.</p>
                  </div>
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">Medical Resident</h3>
                    <p className="text-sm text-gray-500">2015 - 2018</p>
                    <p className="mt-2 text-gray-600">Completed residency with focus on oncology.</p>
                  </div>
                </div>
              </motion.div>
            )}

            {selectedTab === 'reviews' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="space-y-6"
              >
                <div className="space-y-8">
                  {[1, 2, 3].map((review) => (
                    <div key={review} className="border-b border-gray-200 pb-6 last:border-b-0">
                      <div className="flex items-center">
                        <img
                          className="h-10 w-10 rounded-full"
                          src={`https://i.pravatar.cc/40?img=${review}`}
                          alt="Reviewer"
                        />
                        <div className="ml-4">
                          <h4 className="text-sm font-medium text-gray-900">Patient {review}</h4>
                          <div className="mt-1 flex items-center">
                            <span className="text-yellow-400">★★★★★</span>
                            <span className="ml-2 text-sm text-gray-500">2 months ago</span>
                          </div>
                        </div>
                      </div>
                      <p className="mt-4 text-gray-600">
                        Excellent doctor with great attention to detail. Very knowledgeable and caring.
                      </p>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {selectedTab === 'schedule' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="space-y-6"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-gray-50 rounded-lg p-6">
                    <h4 className="text-base font-medium text-gray-900 mb-4">Working Hours</h4>
                    <ul className="space-y-3">
                      <li className="flex justify-between text-gray-600">
                        <span>Monday - Friday</span>
                        <span>9:00 AM - 5:00 PM</span>
                      </li>
                      <li className="flex justify-between text-gray-600">
                        <span>Saturday</span>
                        <span>9:00 AM - 2:00 PM</span>
                      </li>
                      <li className="flex justify-between text-gray-600">
                        <span>Sunday</span>
                        <span>Closed</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-6">
                    <h4 className="text-base font-medium text-gray-900 mb-4">Next Available Slots</h4>
                    <div className="space-y-3">
                      <button className="w-full py-2 px-4 rounded-lg bg-white border border-gray-200 text-gray-700 hover:border-indigo-500 transition-colors">
                        Today at 2:00 PM
                      </button>
                      <button className="w-full py-2 px-4 rounded-lg bg-white border border-gray-200 text-gray-700 hover:border-indigo-500 transition-colors">
                        Tomorrow at 10:00 AM
                      </button>
                      <button className="w-full py-2 px-4 rounded-lg bg-white border border-gray-200 text-gray-700 hover:border-indigo-500 transition-colors">
                        Tomorrow at 3:00 PM
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DoctorProfile; 