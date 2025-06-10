import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();
const BASE_URL = 'https://tumortraker12.runasp.net/api/Account';

export const useAuth = () => {
  return useContext(AuthContext);
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Initialize user from localStorage
  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    const token = localStorage.getItem('token');
    
    if (storedUser && token) {
      try {
        const userData = JSON.parse(storedUser);
        setUser(userData);
      } catch (err) {
        console.error('Error parsing stored user:', err);
        localStorage.removeItem('user');
        localStorage.removeItem('token');
      }
    }
    setLoading(false);
  }, []);

  const registerDoctor = async (formData) => {
    try {
      setLoading(true);
      setError(null);
      
      // Create FormData object for file upload
      const data = new FormData();
      Object.keys(formData).forEach(key => {
        if (key === 'photo' && formData[key]) {
          data.append('Photo', formData[key]);
        } else {
          data.append(key, formData[key]);
        }
      });

      const response = await fetch(`${BASE_URL}/DoctorRegister`, {
        method: 'POST',
        body: data
      });

      const result = await response.json();
      
      if (!response.ok) {
        // Handle email already exists error
        if (result.message === "Email Is Already in Used") {
          throw new Error("This email is already registered");
        }
        
        // Handle validation errors from ModelState
        if (result.errors && Array.isArray(result.errors)) {
          throw new Error(result.errors.join(', '));
        }
        
        // Handle other API errors
        throw new Error(result.message || 'Registration failed');
      }

      // Store user data based on the UserDTO response
      const userData = {
        id: result.id,
        userName: result.userName,
        firstName: formData.firstName,
        lastName: formData.lastName,
        role: result.role,
        email: result.email,
        photoURL: result.photoURL,
        token: result.token
      };

      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
      localStorage.setItem('token', result.token);
      return userData;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const registerPatient = async (formData) => {
    try {
      setLoading(true);
      setError(null);
      
      // Create FormData object for file upload
      const data = new FormData();
      Object.keys(formData).forEach(key => {
        if (key === 'photo' && formData[key]) {
          data.append('Photo', formData[key]);
        } else {
          data.append(key, formData[key]);
        }
      });

      const response = await fetch(`${BASE_URL}/PatientRegister`, {
        method: 'POST',
        body: data
      });

      const result = await response.json();
      
      if (!response.ok) {
        if (result.errors) {
          // Handle different error formats
          if (Array.isArray(result.errors)) {
            throw new Error(result.errors.join(', '));
          } else if (typeof result.errors === 'object') {
            // If errors is an object with error messages
            const errorMessages = Object.values(result.errors)
              .flat()
              .filter(error => error)
              .join(', ');
            throw new Error(errorMessages || 'Registration failed');
          } else if (typeof result.errors === 'string') {
            throw new Error(result.errors);
          }
        }
        throw new Error(result.message || 'Registration failed');
      }

      setUser(result);
      localStorage.setItem('token', result.token);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${BASE_URL}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          email, 
          password 
        }),
      });

      const result = await response.json();
      
      if (!response.ok) {
        if (response.status === 401) {
          throw new Error(result.message);
        }
        if (result.errors) {
          throw new Error(result.errors.join(', '));
        }
        throw new Error(result.message || 'Login failed');
      }

      // Store user data based on the UserDTO response
      const userData = {
        id: result.id,
        userName: result.userName,
        firstName: result.firstName,
        lastName: result.lastName,
        role: result.role,
        email: result.email,
        photoURL: result.photoURL,
        token: result.token
      };

      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
      localStorage.setItem('token', result.token);
      localStorage.setItem('patientId', result.id);
      return userData;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    localStorage.removeItem('patientId');
  };

  const value = {
    user,
    loading,
    error,
    login,
    logout,
    registerDoctor,
    registerPatient
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}; 