import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Register = () => {
  const [formData, setFormData] = useState({
    userType: 'patient', // 'patient' or 'doctor'
    userName: '',
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    age: '',
    websiteURL: '',
    type: '',
    departmentId: '',
    photo: null
  });

  const [errors, setErrors] = useState({});
  const { registerDoctor, registerPatient, loading, error } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value, type, files } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'file' ? files[0] : value
    }));
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    if (!formData.userName.trim()) newErrors.userName = 'Username is required';
    if (!formData.firstName.trim()) newErrors.firstName = 'First name is required';
    if (!formData.lastName.trim()) newErrors.lastName = 'Last name is required';
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }
    if (!formData.password) {
      newErrors.password = 'Password is required';
    }
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }
    if (!formData.age) newErrors.age = 'Age is required';
    if (!formData.type) newErrors.type = 'Type is required';
    
    if (formData.userType === 'doctor' && !formData.departmentId) {
      newErrors.departmentId = 'Department is required';
    }

    return newErrors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const validationErrors = validateForm();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    try {
      setErrors({}); // Clear any previous errors
      const registrationData = {
        userName: formData.userName,
        firstName: formData.firstName,
        lastName: formData.lastName,
        email: formData.email,
        password: formData.password,
        age: parseInt(formData.age),
        websiteURL: formData.websiteURL || '',
        type: formData.type,
        photo: formData.photo
      };

      if (formData.userType === 'doctor') {
        registrationData.departmentId = parseInt(formData.departmentId);
        const userData = await registerDoctor(registrationData);
        // Redirect doctor to their profile page
        navigate('/my-profile');
      } else {
        await registerPatient(registrationData);
        // Redirect patient to home page
        navigate('/');
      }

    } catch (err) {
      // Handle specific error messages
      if (err.message.includes("Email is already registered")) {
        setErrors(prev => ({ ...prev, email: "This email is already registered" }));
      } else {
        // Display the error at the top of the form
        setErrors(prev => ({ 
          ...prev, 
          general: err.message 
        }));
      }
      console.error('Registration failed:', err);
    }
  };

  const inputClasses = (fieldName) => `
    appearance-none relative block w-full px-4 py-3 border 
    ${errors[fieldName] ? 'border-red-500' : 'border-gray-300'}
    rounded-lg placeholder-gray-500 text-gray-900 
    focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent 
    transition duration-150 ease-in-out sm:text-sm
  `;

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-2xl w-full bg-white rounded-xl shadow-2xl p-8">
        <h2 className="text-center text-3xl font-bold text-gray-900 mb-8">
          Create your account
        </h2>

        {/* Display general error if any */}
        {errors.general && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-600">{errors.general}</p>
          </div>
        )}

        <form className="space-y-6" onSubmit={handleSubmit}>
          {/* User Type Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-1">Register as</label>
            <select
              name="userType"
              value={formData.userType}
              onChange={handleChange}
              className={inputClasses('userType')}
            >
              <option value="patient">Patient</option>
              <option value="doctor">Doctor</option>
            </select>
          </div>

          {/* Basic Information Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
              <input
                name="userName"
                type="text"
                required
                className={inputClasses('userName')}
                placeholder="Enter username"
                value={formData.userName}
                onChange={handleChange}
              />
              {errors.userName && (
                <p className="mt-1 text-sm text-red-500">{errors.userName}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
              <input
                name="email"
                type="email"
                required
                className={inputClasses('email')}
                placeholder="Enter email"
                value={formData.email}
                onChange={handleChange}
              />
              {errors.email && (
                <p className="mt-1 text-sm text-red-500">{errors.email}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">First Name</label>
              <input
                name="firstName"
                type="text"
                required
                className={inputClasses('firstName')}
                placeholder="Enter first name"
                value={formData.firstName}
                onChange={handleChange}
              />
              {errors.firstName && (
                <p className="mt-1 text-sm text-red-500">{errors.firstName}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Last Name</label>
              <input
                name="lastName"
                type="text"
                required
                className={inputClasses('lastName')}
                placeholder="Enter last name"
                value={formData.lastName}
                onChange={handleChange}
              />
              {errors.lastName && (
                <p className="mt-1 text-sm text-red-500">{errors.lastName}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
              <input
                name="password"
                type="password"
                required
                className={inputClasses('password')}
                placeholder="Enter password"
                value={formData.password}
                onChange={handleChange}
              />
              {errors.password && (
                <p className="mt-1 text-sm text-red-500">{errors.password}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
              <input
                name="confirmPassword"
                type="password"
                required
                className={inputClasses('confirmPassword')}
                placeholder="Confirm password"
                value={formData.confirmPassword}
                onChange={handleChange}
              />
              {errors.confirmPassword && (
                <p className="mt-1 text-sm text-red-500">{errors.confirmPassword}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
              <input
                name="age"
                type="number"
                required
                className={inputClasses('age')}
                placeholder="Enter age"
                value={formData.age}
                onChange={handleChange}
              />
              {errors.age && (
                <p className="mt-1 text-sm text-red-500">{errors.age}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
              <input
                name="type"
                type="text"
                required
                className={inputClasses('type')}
                placeholder="Enter type"
                value={formData.type}
                onChange={handleChange}
              />
              {errors.type && (
                <p className="mt-1 text-sm text-red-500">{errors.type}</p>
              )}
            </div>
          </div>

          {/* Optional Fields */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Website URL (optional)</label>
              <input
                name="websiteURL"
                type="url"
                className={inputClasses('websiteURL')}
                placeholder="Enter website URL"
                value={formData.websiteURL}
                onChange={handleChange}
              />
            </div>

            {formData.userType === 'doctor' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Department ID</label>
                <input
                  name="departmentId"
                  type="number"
                  required
                  className={inputClasses('departmentId')}
                  placeholder="Enter department ID"
                  value={formData.departmentId}
                  onChange={handleChange}
                />
                {errors.departmentId && (
                  <p className="mt-1 text-sm text-red-500">{errors.departmentId}</p>
                )}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Profile Photo</label>
              <input
                name="photo"
                type="file"
                accept="image/*"
                onChange={handleChange}
                className="mt-1 block w-full py-2 px-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              />
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Creating account...
                </>
              ) : (
                'Create account'
              )}
            </button>
          </div>

          <div className="text-center">
            <p className="text-sm text-gray-600">
              Already have an account?{' '}
              <Link
                to="/login"
                className="font-medium text-indigo-600 hover:text-indigo-500 transition duration-150 ease-in-out"
              >
                Sign in
              </Link>
            </p>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Register; 