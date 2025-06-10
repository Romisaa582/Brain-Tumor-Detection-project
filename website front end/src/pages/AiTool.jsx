import React, { useState, useRef, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { useReactToPrint } from 'react-to-print';

const PrintableReport = React.forwardRef(({ result, user, previewUrl, formatDate, formatProbability, getHighestProbabilityClass }, ref) => {
  if (!result || !user) return null;
  
  return (
    <div ref={ref} className="print-content p-8 bg-white">
      {/* Report Header */}
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900">Brain Tumor Detection Report</h2>
        <p className="text-gray-600 mt-2">Generated on {formatDate(new Date())}</p>
      </div>

      {/* Patient Information */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold mb-4 border-b pb-2">Patient Information</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-gray-600">Name: {user?.firstName} {user?.lastName}</p>
            <p className="text-gray-600">Email: {user?.email}</p>
            <p className="text-gray-600">ID: {user?.id}</p>
          </div>
          {user?.photoURL && (
            <div className="flex justify-end">
              <img 
                src={user.photoURL} 
                alt="Patient" 
                className="w-24 h-24 rounded-full object-cover"
              />
            </div>
          )}
        </div>
      </div>

      {/* MRI Scan Image */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold mb-4 border-b pb-2">MRI Scan</h3>
        <div className="flex justify-center">
          <img
            src={previewUrl}
            alt="MRI Scan"
            className="max-h-64 object-contain"
          />
        </div>
      </div>

      {/* Analysis Results */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold mb-4 border-b pb-2">Analysis Results</h3>
        <div className="space-y-4">
          <div className="mb-4">
            <p className="text-lg font-medium">
              Primary Detection: {' '}
              <span className={
                getHighestProbabilityClass(result) === 'No Tumor'
                  ? 'text-green-600'
                  : 'text-red-600'
              }>
                {getHighestProbabilityClass(result)}
              </span>
            </p>
          </div>
          
          <div className="space-y-3">
            {Object.entries(result).map(([type, probability]) => (
              <div key={type}>
                <div className="flex justify-between mb-1">
                  <span className="font-medium">{type}:</span>
                  <span>{formatProbability(probability)}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-indigo-600 h-2 rounded-full"
                    style={{ width: `${probability * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {getHighestProbabilityClass(result) !== 'No Tumor' && (
        <div>
          <h3 className="text-xl font-semibold mb-4 border-b pb-2">Recommendations</h3>
          <ul className="list-disc list-inside space-y-2">
            <li>Schedule an immediate consultation with a specialist</li>
            <li>Bring your MRI scans and AI analysis results to your appointment</li>
            <li>Consider getting a second opinion from another specialist</li>
          </ul>
        </div>
      )}

      {/* Footer */}
      <div className="mt-8 pt-4 border-t text-sm text-gray-500">
        <p>This report was generated automatically by the AI Brain Tumor Detection System.</p>
        <p>Please consult with a healthcare professional for proper medical diagnosis.</p>
      </div>
    </div>
  );
});

const AiTool = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [printError, setPrintError] = useState(null);
  const { user } = useAuth();
  const reportRef = useRef(null);

  const formatDate = (date) => {
    try {
      return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (error) {
      console.error('Date formatting error:', error);
      return 'Date not available';
    }
  };

  const formatProbability = (value) => {
    try {
      return (value * 100).toFixed(2) + '%';
    } catch (error) {
      console.error('Probability formatting error:', error);
      return 'N/A';
    }
  };

  const getHighestProbabilityClass = (results) => {
    try {
      if (!results || typeof results !== 'object') return 'N/A';
      return Object.entries(results).reduce((a, b) => (a[1] > b[1] ? a : b), ['N/A', 0])[0];
    } catch (error) {
      console.error('Error getting highest probability:', error);
      return 'N/A';
    }
  };

  React.useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @media print {
        body * {
          visibility: hidden;
        }
        .print-content,
        .print-content * {
          visibility: visible !important;
        }
        .print-content {
          position: fixed;
          left: 0;
          top: 0;
          width: 100%;
        }
      }
    `;
    document.head.appendChild(style);
    return () => {
      if (style.parentNode) {
        document.head.removeChild(style);
      }
    };
  }, []);

  const handlePrintAction = useCallback(() => {
    if (!result || !user) {
      setPrintError('Cannot print: Missing required data');
      return;
    }

    try {
      handlePrint();
    } catch (error) {
      console.error('Print error:', error);
      setPrintError('Failed to print. Please try again.');
    }
  }, [result, user]);

  const handlePrint = useReactToPrint({
    content: useCallback(() => reportRef.current, []),
    onBeforeGetContent: useCallback(() => {
      setPrintError(null);
      console.log('Preparing to print...');
      return Promise.resolve();
    }, []),
    onPrintError: useCallback((error) => {
      console.error('Print failed:', error);
      setPrintError('Failed to print. Please try again.');
    }, []),
    onAfterPrint: useCallback(() => {
      console.log('Print completed or cancelled');
      setPrintError(null);
    }, []),
    removeAfterPrint: true
  });

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedImage) return;

    setLoading(true);
    setPrintError(null);
    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch('https://samxrashed-braintumorclassification.hf.space/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze image');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error analyzing image:', error);
      setResult({ error: 'Failed to analyze image. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <div className="max-w-4xl mx-auto">
        {/* Header Section */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4 relative inline-block">
            AI Brain Tumor Detection
            <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 to-blue-500 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left"></div>
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload a brain MRI scan to detect and classify potential tumors using advanced AI technology.
            Our system can identify Glioma, Meningioma, Pituitary tumors, or confirm No Tumor presence.
          </p>
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-2xl shadow-xl p-8 backdrop-blur-lg bg-opacity-90 transition-all duration-300 hover:shadow-2xl">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Image Upload Section */}
            <div className="space-y-4">
              <label className="block text-lg font-medium text-gray-700 mb-4">
                Upload MRI Scan Image
              </label>
              
              <div 
                className="flex justify-center px-6 pt-5 pb-6 border-2 border-dashed rounded-2xl transition-all duration-300 ease-in-out
                  hover:border-indigo-500 hover:bg-indigo-50/30
                  group cursor-pointer relative"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <div className="space-y-1 text-center">
                  <div className="flex flex-col items-center">
                    {previewUrl ? (
                      <div className="relative w-full max-w-md group">
                        <img
                          src={previewUrl}
                          alt="Preview"
                          className="max-h-64 rounded-lg object-contain shadow-lg transition-transform duration-300 group-hover:scale-105"
                        />
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedImage(null);
                            setPreviewUrl(null);
                          }}
                          className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-2 shadow-lg hover:bg-red-600 transition-all duration-300 transform hover:scale-110"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    ) : (
                      <>
                        <div className="transform transition-transform duration-300 group-hover:scale-110">
                          <svg
                            className="mx-auto h-16 w-16 text-gray-400 group-hover:text-indigo-500 transition-colors duration-300"
                            stroke="currentColor"
                            fill="none"
                            viewBox="0 0 48 48"
                          >
                            <path
                              d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </svg>
                        </div>
                        <div className="mt-4">
                          <label className="relative cursor-pointer bg-transparent rounded-md font-medium text-indigo-600 hover:text-indigo-500 transition-colors duration-300">
                            <span className="text-base">Upload a file</span>
                            <input
                              type="file"
                              className="sr-only"
                              accept="image/*"
                              onChange={handleImageChange}
                            />
                          </label>
                          <p className="pl-1 text-base text-gray-500 inline">or drag and drop</p>
                        </div>
                        <p className="text-sm text-gray-500 mt-2">PNG, JPG, GIF up to 10MB</p>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <div>
              <button
                type="submit"
                disabled={!selectedImage || loading}
                className={`w-full flex justify-center items-center py-4 px-6 border border-transparent rounded-xl text-base font-medium shadow-lg
                  transition-all duration-300 ease-in-out transform hover:-translate-y-1
                  ${!selectedImage || loading 
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : 'bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white'
                  }`}
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing Image...
                  </>
                ) : (
                  'Analyze Image'
                )}
              </button>
            </div>
          </form>

          {/* Results Section */}
          {result && !result.error && (
            <>
              {/* Printable Report */}
              <div style={{ display: 'none' }}>
                <PrintableReport
                  ref={reportRef}
                  result={result}
                  user={user}
                  previewUrl={previewUrl}
                  formatDate={formatDate}
                  formatProbability={formatProbability}
                  getHighestProbabilityClass={getHighestProbabilityClass}
                />
              </div>

              {/* Visible Results */}
              <div className="mt-8 p-8 bg-gradient-to-br from-white to-blue-50 rounded-2xl shadow-lg transition-all duration-300">
                <div className="flex justify-between items-center mb-6 pb-2 border-b border-gray-200">
                  <h3 className="text-xl font-semibold text-gray-900">Analysis Results</h3>
                  <button
                    onClick={handlePrintAction}
                    className="flex items-center px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                    disabled={loading || !result || !user}
                  >
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
                    </svg>
                    Print Report
                  </button>
                </div>

                {printError && (
                  <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">
                    {printError}
                  </div>
                )}

                <div className="space-y-6">
                  <div className="bg-white p-6 rounded-xl shadow-sm">
                    <h4 className="text-lg font-semibold text-gray-900 mb-4">
                      Detection Result: {' '}
                      <span className={`${
                        getHighestProbabilityClass(result) === 'No Tumor' 
                          ? 'text-green-600' 
                          : 'text-red-600'
                      }`}>
                        {getHighestProbabilityClass(result)}
                      </span>
                    </h4>
                    
                    <div className="space-y-3">
                      {Object.entries(result).map(([type, probability]) => (
                        <div key={type} className="relative">
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium text-gray-700">{type}</span>
                            <span className="text-sm font-medium text-gray-700">
                              {formatProbability(probability)}
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div
                              className="bg-indigo-600 h-2.5 rounded-full transition-all duration-500"
                              style={{ width: `${probability * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {getHighestProbabilityClass(result) !== 'No Tumor' && (
                    <div className="bg-white p-6 rounded-xl shadow-sm">
                      <p className="font-semibold text-gray-900 mb-4">Recommendations</p>
                      <ul className="space-y-3">
                        <li className="flex items-center text-gray-700">
                          <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          Schedule an immediate consultation with a specialist
                        </li>
                        <li className="flex items-center text-gray-700">
                          <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          Bring your MRI scans and AI analysis results to your appointment
                        </li>
                        <li className="flex items-center text-gray-700">
                          <svg className="w-5 h-5 text-indigo-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          Consider getting a second opinion from another specialist
                        </li>
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default AiTool; 