import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { specialityData } from '../assets/assets';
import Chat from '../components/Chat';

const Doctor = () => {
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeChatDoctor, setActiveChatDoctor] = useState(null);
  const navigate = useNavigate();
  const { speciality } = useParams();

  useEffect(() => {
    fetchDoctors();
  }, []);

  const fetchDoctors = async () => {
    try {
      setLoading(true);
      const response = await fetch('https://tumortraker12.runasp.net/api/Account/GetAllDoctors');
      if (!response.ok) {
        throw new Error('Failed to fetch doctors');
      }
      const data = await response.json();
      setDoctors(data);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching doctors:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredDoctors = speciality
    ? doctors.filter(doctor => doctor.type.toLowerCase() === speciality.toLowerCase())
    : doctors;

  const handleCloseChat = () => {
    setActiveChatDoctor(null);
  };

  if (loading) {
    return <div className="text-center p-4">Loading doctors...</div>;
  }

  if (error) {
    return (
      <div className="text-center p-4">
        <p className="text-red-600">Error: {error}</p>
        <button onClick={fetchDoctors} className="text-blue-600">
          Try again
        </button>
      </div>
    );
  }

  return (
    <div className="p-4">
      {/* Speciality Filter */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-center mb-4">
          {speciality ? `${speciality} Specialists` : 'All Doctors'}
        </h1>
        <div className="flex flex-wrap justify-center gap-2">
          <button
            onClick={() => navigate('/doctors')}
            className={`px-3 py-1 border rounded ${
              !speciality ? 'bg-blue-600 text-white' : 'bg-white'
            }`}
          >
            All
          </button>
          {specialityData.map((item) => (
            <button
              key={item.speciality}
              onClick={() => navigate(`/doctors/${item.speciality}`)}
              className={`px-3 py-1 border rounded ${
                speciality === item.speciality ? 'bg-blue-600 text-white' : 'bg-white'
              }`}
            >
              {item.speciality}
            </button>
          ))}
        </div>
      </div>

      {/* Doctors Grid */}
      <div className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 p-4">
        {filteredDoctors.map((doctor) => (
          <div key={doctor.id} className="bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow duration-300 overflow-hidden">
            {/* Doctor Image */}
            <div className="relative h-[300px] overflow-hidden bg-gray-100">
              {doctor.photoURL ? (
                <img
                  src={doctor.photoURL}
                  alt={`${doctor.firstName} ${doctor.lastName}`}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2NjYyI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgM2MyLjY3IDAgNC41NCAyLjU4IDQuNSA1LjQ1IDAgMi45NS0xLjgzIDUuNDUtNC41IDUuNDVzLTQuNS0yLjUtNC41LTUuNDVjLS4wMy0yLjg3IDEuODQtNS40NSA0LjUtNS40NXptNy41IDE1LjVjLTEuMjUgMS42OC0zLjE3IDMtNS41IDMuNXMtNC43NS4zLTYuNS0uNWMtMS4yOC0uNi0yLjI3LTEuNy0yLjkzLTIuOTggMS4zNC0xLjI4IDUuNzQtMi4yOCA4LjkzLTIuMjhzNy41OSAxIDguOTMgMi4yOGMtLjY2IDEuMjgtMS42NSAyLjM4LTIuOTMgMi45OHoiLz48L3N2Zz4=';
                  }}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-24 h-24 text-gray-300" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c2.67 0 4.54 2.58 4.5 5.45 0 2.95-1.83 5.45-4.5 5.45s-4.5-2.5-4.5-5.45c-.03-2.87 1.84-5.45 4.5-5.45zm7.5 15.5c-1.25 1.68-3.17 3-5.5 3.5s-4.75.3-6.5-.5c-1.28-.6-2.27-1.7-2.93-2.98 1.34-1.28 5.74-2.28 8.93-2.28s7.59 1 8.93 2.28c-.66 1.28-1.65 2.38-2.93 2.98z" />
                  </svg>
                </div>
              )}
            </div>

            {/* Doctor Info */}
            <div className="p-4">
              <div className="mb-3">
                <h3 className="text-lg font-bold text-gray-800">
                  Dr. {doctor.firstName} {doctor.lastName}
                </h3>
                <div className="flex items-center gap-2 mt-2">
                  <span className={`inline-block w-2 h-2 rounded-full ${
                    doctor.type === 'Admin' ? 'bg-green-500' : 
                    doctor.gender === 'Male' ? 'bg-blue-500' : 'bg-pink-500'
                  }`}></span>
                  <span className="text-sm text-gray-600">{doctor.type}</span>
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  <span>Specialization: {doctor.specialization || 'General Medicine'}</span>
                </div>
              </div>

              {/* Appointment Fee and Chat */}
              <div className="mt-6 bg-blue-50 p-4 rounded-xl">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">Appointment Fee:</h3>
                  <span className="text-2xl font-bold text-blue-600">$50</span>
                </div>
                <button
                  onClick={() => setActiveChatDoctor(doctor)}
                  className="w-full flex items-center justify-center gap-2 bg-white hover:bg-blue-600 text-blue-600 hover:text-white py-3 rounded-lg transition-all duration-300 border-2 border-blue-600"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  Chat with Doctor
                </button>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-3 mt-4 pt-3 border-t">
                {doctor.websiteURL && (
                  <a
                    href={doctor.websiteURL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                    </svg>
                    Website
                  </a>
                )}
                <button
                  onClick={() => navigate(`/my-profile?id=${doctor.id}`)}
                  className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                  View Profile
                </button>
                <button
                  onClick={() => navigate(`/appointment/${doctor.id}`)}
                  className="ml-auto text-sm bg-blue-600 text-white px-4 py-1.5 rounded-full hover:bg-blue-700 transition-colors duration-200 flex items-center gap-1"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  Book
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredDoctors.length === 0 && (
        <div className="text-center text-gray-500 mt-4">
          No doctors found {speciality ? `for ${speciality}` : ''}.
        </div>
      )}

      {/* Chat Window */}
      {activeChatDoctor && (
        <Chat
          doctorId={activeChatDoctor.id}
          doctorName={`Dr. ${activeChatDoctor.firstName} ${activeChatDoctor.lastName}`}
          onClose={handleCloseChat}
        />
      )}
    </div>
  );
};

export default Doctor;