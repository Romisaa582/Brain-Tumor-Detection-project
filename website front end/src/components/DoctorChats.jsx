import { useState, useEffect, useCallback } from 'react';
import { jwtDecode } from 'jwt-decode';
import Chat from './Chat';
import { signalRService } from '../services/signalRService';

const DoctorChats = ({ onClose }) => {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const token = localStorage.getItem('token');
  const decodedToken = token ? jwtDecode(token) : null;
  const doctorId = decodedToken?.Id;

  const formatTime = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.abs(now - date) / 36e5;

    if (diffInHours < 24) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffInHours < 48) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString();
    }
  };

  const fetchWithAuth = async (url, options = {}) => {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          ...options.headers
        }
      });

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Unauthorized. Please log in again.');
        }
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.message || `Error: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  };

  const loadPatients = useCallback(async () => {
    try {
      setError(null);
      
      // Get contact users
      const users = await fetchWithAuth('https://tumortraker12.runasp.net/api/Chat/GetContactUsers');
      
      // Get chat history for each patient
      const patientsWithMessages = await Promise.all(
        users.map(async (patient) => {
          try {
            const messages = await fetchWithAuth(
              `https://tumortraker12.runasp.net/api/Chat/GetChat?receiverId=${patient.id}`
            );
            
            const lastMessage = messages[messages.length - 1];
            return {
              id: patient.id,
              firstName: patient.firstName,
              lastName: patient.lastName,
              userName: patient.userName,
              email: patient.email,
              photoURL: patient.photoURL,
              type: patient.type,
              lastMessage: lastMessage?.message || '',
              lastMessageTime: lastMessage?.timestamp,
              lastMessageSender: lastMessage?.senderId,
              unreadCount: messages.filter(m => m.senderId !== doctorId && !m.isRead).length
            };
          } catch (error) {
            console.warn(`Failed to load chat history for patient ${patient.id}:`, error);
            return {
              id: patient.id,
              firstName: patient.firstName,
              lastName: patient.lastName,
              userName: patient.userName,
              email: patient.email,
              photoURL: patient.photoURL,
              type: patient.type,
              lastMessage: '',
              unreadCount: 0
            };
          }
        })
      );

      // Sort patients by latest message time
      const sortedPatients = patientsWithMessages.sort((a, b) => {
        const dateA = a.lastMessageTime ? new Date(a.lastMessageTime) : new Date(0);
        const dateB = b.lastMessageTime ? new Date(b.lastMessageTime) : new Date(0);
        return dateB - dateA;
      });

      setPatients(sortedPatients);
    } catch (error) {
      console.error('Error loading patients:', error);
      if (error.message.includes('Unauthorized')) {
        setError('Session expired. Please log in again.');
      } else {
        setError('Failed to load messages. Please try again in a moment.');
      }
    } finally {
      setLoading(false);
    }
  }, [token, doctorId]);

  const handleNewMessage = useCallback((senderId, message) => {
    setPatients(prevPatients => {
      const updatedPatients = prevPatients.map(patient => {
        if (patient.id === senderId) {
          return {
            ...patient,
            lastMessage: message,
            lastMessageTime: new Date().toISOString(),
            lastMessageSender: senderId,
            unreadCount: (patient.unreadCount || 0) + 1
          };
        }
        return patient;
      });

      // Move the patient with new message to top
      const patientIndex = updatedPatients.findIndex(p => p.id === senderId);
      if (patientIndex > 0) {
        const [patient] = updatedPatients.splice(patientIndex, 1);
        updatedPatients.unshift(patient);
      }

      return updatedPatients;
    });
  }, []);

  useEffect(() => {
    let mounted = true;
    let retryCount = 0;
    const maxRetries = 3;

    const initialize = async () => {
      while (retryCount < maxRetries && mounted) {
        try {
          await loadPatients();
          break;
        } catch (error) {
          retryCount++;
          if (retryCount < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, 2000 * retryCount));
          }
        }
      }

      if (mounted) {
        // Connect to SignalR
        const connected = await signalRService.startConnection();
        if (connected) {
          const removeListener = signalRService.addMessageListener(handleNewMessage);
          return () => {
            removeListener();
            signalRService.stopConnection();
          };
        }
      }
    };

    initialize();

    return () => {
      mounted = false;
    };
  }, [loadPatients, handleNewMessage]);

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white p-6 rounded-lg shadow-xl">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading messages...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white p-6 rounded-lg shadow-xl max-w-md w-full">
          <div className="text-red-500 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-center text-gray-700">{error}</p>
          <div className="mt-4 flex justify-center gap-2">
            <button
              onClick={loadPatients}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              Try Again
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (selectedPatient) {
    return (
      <Chat
        doctorId={doctorId}
        patientId={selectedPatient.id}
        patientName={`${selectedPatient.firstName} ${selectedPatient.lastName}`}
        onClose={() => setSelectedPatient(null)}
        onMessageSent={loadPatients}
      />
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
        <div className="p-4 border-b flex items-center justify-between bg-indigo-600 rounded-t-lg">
          <h2 className="text-xl font-semibold text-white">Messages</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-indigo-700 rounded-full transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="p-4 max-h-[70vh] overflow-y-auto">
          {patients.length === 0 ? (
            <div className="text-center py-8">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <p className="text-gray-500 text-lg">No messages yet</p>
              <p className="text-gray-400 text-sm mt-2">When patients send you messages, they'll appear here</p>
            </div>
          ) : (
            <div className="space-y-2">
              {patients.map((patient) => (
                <div
                  key={patient.id}
                  onClick={() => setSelectedPatient(patient)}
                  className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors border border-gray-100"
                >
                  <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0">
                    {patient.photoURL ? (
                      <img
                        src={patient.photoURL}
                        alt={patient.firstName}
                        className="w-12 h-12 rounded-full object-cover"
                        onError={(e) => {
                          e.target.src = 'https://via.placeholder.com/40?text=P';
                        }}
                      />
                    ) : (
                      <span className="text-indigo-600 font-semibold text-lg">
                        {patient.firstName[0]}
                      </span>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-start">
                      <h3 className="font-medium text-gray-900 truncate">
                        {patient.firstName} {patient.lastName}
                      </h3>
                      {patient.lastMessageTime && (
                        <span className="text-xs text-gray-500 ml-2">
                          {formatTime(patient.lastMessageTime)}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      {patient.lastMessageSender === doctorId && (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                      <p className="text-sm text-gray-500 truncate">
                        {patient.lastMessage || 'No messages yet'}
                      </p>
                    </div>
                  </div>
                  {patient.unreadCount > 0 && (
                    <span className="ml-2 bg-indigo-600 text-white text-xs font-medium px-2 py-1 rounded-full">
                      {patient.unreadCount}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DoctorChats; 