import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Link, useSearchParams, useParams, useNavigate } from 'react-router-dom';

const MyProfile = () => {
  const { user } = useAuth();
  const [searchParams] = useSearchParams();
  const [activeTab, setActiveTab] = useState('profile');
  const { patientId } = useParams();
  const [patientData, setPatientData] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const tab = searchParams.get('tab');
    if (tab && ['profile', 'appointments', 'medical', 'settings'].includes(tab)) {
      setActiveTab(tab);
    }
  }, [searchParams]);

  useEffect(() => {
    const fetchPatientData = async () => {
      if (patientId && user?.role === 'Doctor') {
        try {
          const token = localStorage.getItem('token');
          const response = await fetch(`https://tumortraker12.runasp.net/api/Patient/${patientId}`, {
            headers: {
              'Authorization': `Bearer ${token}`
            }
          });

          if (!response.ok) {
            throw new Error('Failed to fetch patient data');
          }

          const data = await response.json();
          setPatientData(data);
        } catch (error) {
          console.error('Error fetching patient data:', error);
          navigate('/my-patients');
        }
      }
    };

    fetchPatientData();
  }, [patientId, user?.role, navigate]);

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900">Please log in to view profile</h2>
        </div>
      </div>
    );
  }

  // If we're viewing a patient's profile as a doctor
  const profileData = patientId && patientData ? patientData : user;
  const isViewingPatient = patientId && user?.role === 'Doctor';

  const tabs = [
    { id: 'profile', label: 'Profile', icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
      </svg>
    )},
    { id: 'medical', label: 'Medical History', icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    )}
  ];

  // Only show appointments and settings tabs for own profile
  if (!isViewingPatient) {
    tabs.push(
      { id: 'appointments', label: 'Appointments', icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      )},
      { id: 'settings', label: 'Settings', icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      )}
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Profile Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-blue-600 h-60">
        {isViewingPatient && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-4">
            <button
              onClick={() => navigate('/my-patients')}
              className="flex items-center text-white hover:text-indigo-100 transition-colors"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to My Patients
            </button>
          </div>
        )}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-full flex items-end">
          <div className="pb-6 flex items-center space-x-5">
            <div className="relative">
              <img
                src={profileData.photoURL || 'https://via.placeholder.com/150?text=User'}
                alt="Profile"
                className="w-32 h-32 rounded-full border-4 border-white shadow-lg object-cover"
              />
              <div className="absolute bottom-2 right-2 w-4 h-4 bg-green-400 border-2 border-white rounded-full"></div>
            </div>
            <div className="text-white">
              <h1 className="text-3xl font-bold">{profileData.firstName} {profileData.lastName}</h1>
              <div className="flex items-center mt-1 space-x-3">
                <p className="text-indigo-100">@{profileData.userName || 'user'}</p>
                <span className="text-indigo-200">•</span>
                <p className="text-indigo-100">{isViewingPatient ? 'Patient' : profileData.role}</p>
                {isViewingPatient && (
                  <>
                    <span className="text-indigo-200">•</span>
                    <p className="text-indigo-100">Viewing as Doctor</p>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 border-b-2 px-1 ${
                  activeTab === tab.id
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            {/* Basic Info Card */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Basic Information</h2>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-500">Full Name</label>
                  <p className="mt-1 text-gray-900">{profileData.firstName} {profileData.lastName}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Email</label>
                  <p className="mt-1 text-gray-900">{profileData.email}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Username</label>
                  <p className="mt-1 text-gray-900">@{profileData.userName || 'user'}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Role</label>
                  <p className="mt-1 text-gray-900">{isViewingPatient ? 'Patient' : profileData.role}</p>
                </div>
              </div>
            </div>

            {/* Contact Info Card */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Contact Information</h2>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-500">Phone Number</label>
                  <p className="mt-1 text-gray-900">{profileData.phone || 'Not provided'}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Address</label>
                  <p className="mt-1 text-gray-900">{profileData.address || 'Not provided'}</p>
                </div>
                {!isViewingPatient && user.role === 'Doctor' && (
                  <div className="pt-2">
                    <Link
                      to="/chat"
                      className="inline-flex items-center justify-center w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors duration-200"
                    >
                      <svg 
                        className="w-5 h-5 mr-2" 
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                      >
                        <path 
                          strokeLinecap="round" 
                          strokeLinejoin="round" 
                          strokeWidth="2" 
                          d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" 
                        />
                      </svg>
                      Chat with Patients
                    </Link>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="lg:col-span-2">
            {activeTab === 'profile' && (
              <div className="bg-white rounded-lg shadow">
                <div className="p-6 border-b border-gray-200">
                  <h2 className="text-lg font-semibold text-gray-900">Profile Overview</h2>
                  <p className="mt-1 text-sm text-gray-500">
                    {isViewingPatient ? 'Patient profile information and history.' : 'Manage your profile information and preferences.'}
                  </p>
                </div>
                <div className="p-6">
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-sm font-medium text-gray-500">Member Since</h3>
                      <p className="mt-1 text-gray-900">
                        {profileData.createdAt ? new Date(profileData.createdAt).toLocaleDateString() : 'N/A'}
                      </p>
                    </div>
                    {!isViewingPatient && (
                      <div>
                        <h3 className="text-sm font-medium text-gray-500">Last Login</h3>
                        <p className="mt-1 text-gray-900">
                          {profileData.lastLogin ? new Date(profileData.lastLogin).toLocaleString() : 'N/A'}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'medical' && isViewingPatient && (
              <div className="bg-white rounded-lg shadow">
                <div className="p-6 border-b border-gray-200">
                  <h2 className="text-lg font-semibold text-gray-900">Medical History</h2>
                  <p className="mt-1 text-sm text-gray-500">
                    Patient's medical history and records.
                  </p>
                </div>
                <div className="p-6">
                  {/* Add medical history content here */}
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-sm font-medium text-gray-500">Diagnosis</h3>
                      <p className="mt-1 text-gray-900">{patientData?.diagnosis || 'No diagnosis recorded'}</p>
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-gray-500">Treatment Plan</h3>
                      <p className="mt-1 text-gray-900">{patientData?.treatmentPlan || 'No treatment plan recorded'}</p>
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-gray-500">Notes</h3>
                      <p className="mt-1 text-gray-900">{patientData?.notes || 'No notes available'}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {!isViewingPatient && (
              <>
                {activeTab === 'appointments' && (
                  <div className="bg-white rounded-lg shadow">
                    <div className="p-6 border-b border-gray-200">
                      <h2 className="text-lg font-semibold text-gray-900">Your Appointments</h2>
                      <p className="mt-1 text-sm text-gray-500">
                        View and manage your upcoming and past appointments.
                      </p>
                    </div>
                    <div className="p-6">
                      {/* Existing appointments content */}
                    </div>
                  </div>
                )}

                {activeTab === 'settings' && (
                  <div className="bg-white rounded-lg shadow">
                    <div className="p-6 border-b border-gray-200">
                      <h2 className="text-lg font-semibold text-gray-900">Account Settings</h2>
                      <p className="mt-1 text-sm text-gray-500">
                        Manage your account settings and preferences.
                      </p>
                    </div>
                    <div className="p-6">
                      {/* Existing settings content */}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MyProfile;