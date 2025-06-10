import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import LOGO from '../assets/LOGO1.png';
import DoctorChats from './DoctorChats';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [showChats, setShowChats] = useState(false);
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
    } catch (error) {
      console.error('Failed to log out:', error);
    }
  };

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="flex items-center justify-between text-sm py-4 mb-5 border-b border-b-gray-400">
      <img 
        className="w-20 cursor-pointer mt-4" 
        src={LOGO} 
        alt="Logo" 
        onClick={() => navigate('/')}
      />

      {user ? (
        <>
          <ul className="hidden md:flex items-center gap-5 font-medium">
            <Link 
              to="/"
              className="py-1 relative group text-gray-700 hover:text-indigo-600"
            >
              <li>Home</li>
              <div className="absolute bottom-0 left-0 w-full h-0.5 bg-indigo-600 scale-x-0 group-hover:scale-x-100 transition-transform" />
            </Link>

            {user.role === 'Patient' ? (
              <>
                <Link 
                  to="/doctors"
                  className="py-1 relative group text-gray-700 hover:text-indigo-600"
                >
                  <li>All Doctors</li>
                  <div className="absolute bottom-0 left-0 w-full h-0.5 bg-indigo-600 scale-x-0 group-hover:scale-x-100 transition-transform" />
                </Link>

                <Link 
                  to="/ai-tool"
                  className="py-1 relative group text-gray-700 hover:text-indigo-600"
                >
                  <li>AI Tool</li>
                  <div className="absolute bottom-0 left-0 w-full h-0.5 bg-indigo-600 scale-x-0 group-hover:scale-x-100 transition-transform" />
                </Link>
              </>
            ) : user.role === 'Doctor' && (
              <>
                <Link 
                  to="/my-patients"
                  className="py-1 relative group text-gray-700 hover:text-indigo-600"
                >
                  <li>My Patients</li>
                  <div className="absolute bottom-0 left-0 w-full h-0.5 bg-indigo-600 scale-x-0 group-hover:scale-x-100 transition-transform" />
                </Link>
                <Link 
                  to="/schedule"
                  className="py-1 relative group text-gray-700 hover:text-indigo-600"
                >
                  <li>Work Schedule</li>
                  <div className="absolute bottom-0 left-0 w-full h-0.5 bg-indigo-600 scale-x-0 group-hover:scale-x-100 transition-transform" />
                </Link>
                <button
                  onClick={() => setShowChats(true)}
                  className="py-1 relative group text-gray-700 hover:text-indigo-600 flex items-center gap-1"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  Chats
                </button>
              </>
            )}

            <Link 
              to="/about"
              className="py-1 relative group text-gray-700 hover:text-indigo-600"
            >
              <li>About</li>
              <div className="absolute bottom-0 left-0 w-full h-0.5 bg-indigo-600 scale-x-0 group-hover:scale-x-100 transition-transform" />
            </Link>

            <Link 
              to="/contact"
              className="py-1 relative group text-gray-700 hover:text-indigo-600"
            >
              <li>Contact</li>
              <div className="absolute bottom-0 left-0 w-full h-0.5 bg-indigo-600 scale-x-0 group-hover:scale-x-100 transition-transform" />
            </Link>
          </ul>

          <div className="relative">
            <div 
              onClick={toggleDropdown}
              className="flex items-center gap-3 cursor-pointer p-2 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center gap-2">
                <div className="relative">
                  <img 
                    className="w-10 h-10 rounded-full object-cover ring-2 ring-indigo-600 ring-offset-2"
                    src={user.photoURL || 'https://via.placeholder.com/40?text=User'} 
                    alt={user.userName}
                    onError={(e) => {
                      e.target.src = 'https://via.placeholder.com/40?text=User';
                    }}
                  />
                  <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-400 border-2 border-white rounded-full"></div>
                </div>
                <div className="hidden sm:block">
                  <p className="text-gray-900 font-medium leading-tight">
                    {user.firstName} {user.lastName}
                  </p>
                  <p className="text-gray-500 text-xs">@{user.userName || 'user'}</p>
                </div>
              </div>
              <svg 
                className={`w-4 h-4 text-gray-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>

            {/* Enhanced Dropdown Menu */}
            <div className={`absolute right-0 mt-2 w-72 bg-white rounded-xl shadow-lg py-3 transition-all duration-200 transform ${
              isOpen 
                ? 'visible opacity-100 translate-y-0' 
                : 'invisible opacity-0 translate-y-2'
            } z-50`}>
              <div className="px-4 py-3 border-b border-gray-100">
                <div className="flex items-center gap-3">
                  <img 
                    className="w-12 h-12 rounded-full object-cover ring-2 ring-indigo-600 ring-offset-2"
                    src={user.photoURL || 'https://via.placeholder.com/40?text=User'} 
                    alt={user.userName}
                  />
                  <div>
                    <p className="text-sm font-medium text-gray-900">{user.firstName} {user.lastName}</p>
                    <p className="text-xs text-indigo-600 font-medium">@{user.userName || 'user'}</p>
                    <p className="text-xs text-gray-500 truncate">{user.email}</p>
                  </div>
                </div>
              </div>
              
              <div className="py-2">
                <Link
                  to="/my-profile"
                  className="flex items-center gap-3 w-full px-4 py-2 text-gray-700 hover:bg-indigo-50 hover:text-indigo-600 transition-colors"
                  onClick={() => setIsOpen(false)}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                  My Profile
                </Link>

                {user.role === 'Patient' && (
                  <Link
                    to="/my-appointments"
                    className="flex items-center gap-3 w-full px-4 py-2 text-gray-700 hover:bg-indigo-50 hover:text-indigo-600 transition-colors"
                    onClick={() => setIsOpen(false)}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    My Appointments
                  </Link>
                )}

                <Link
                  to="/my-profile?tab=settings"
                  className="flex items-center gap-3 w-full px-4 py-2 text-gray-700 hover:bg-indigo-50 hover:text-indigo-600 transition-colors"
                  onClick={() => setIsOpen(false)}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  Settings
                </Link>

                {user.role === 'Doctor' && (
                  <Link
                    to="/schedule"
                    className="flex items-center gap-3 w-full px-4 py-2 text-gray-700 hover:bg-indigo-50 hover:text-indigo-600 transition-colors"
                    onClick={() => setIsOpen(false)}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Work Time
                  </Link>
                )}
              </div>

              <div className="border-t border-gray-100 pt-2">
                <button
                  onClick={() => {
                    setIsOpen(false);
                    handleLogout();
                  }}
                  className="flex items-center gap-3 w-full px-4 py-2 text-red-600 hover:bg-red-50 transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                  </svg>
                  Logout
                </button>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="flex items-center gap-4">
          <Link
            to="/login"
            className="py-2 px-4 text-indigo-600 rounded-lg hover:bg-indigo-50 transition-colors"
          >
            Log In
          </Link>
          <Link
            to="/register"
            className="py-2 px-4 text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Sign Up
          </Link>
        </div>
      )}

      {showChats && <DoctorChats onClose={() => setShowChats(false)} />}
    </div>
  );
};

export default Navbar;