import React from 'react';
import { Link } from 'react-router-dom';

const ForYou = () => {
  return (
    <div className="min-h-screen py-8">
      {/* Mental Health Information Section */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center mb-16">
        <h2 className="text-5xl font-bold mb-2 text-gray-900">MENTAL</h2>
        <h3 className="text-3xl font-light mb-6 text-primary">HEALTH</h3>
        <p className="text-lg mb-12 max-w-3xl mx-auto text-gray-600">
          This page is dedicated to mental health, providing essential information to help
          you understand your condition. It serves as a pathway to achieving better
          mental well-being, aiding in faster recovery and a healthier life.
        </p>

        {/* Therapy Session Card */}
        <div className="bg-white rounded-2xl p-8 text-left shadow-lg border border-gray-100">
          <h3 className="text-2xl font-bold mb-6 text-gray-900">Individual Therapy Sessions</h3>
          <p className="text-lg mb-6 text-gray-600">
            Book a One-on-One Session with a Specialist Our individual therapy sessions
            provide personalized support tailored to your unique needs. Schedule a session
            with one of our experienced mental health professionals to start your journey
            towards better mental health.
          </p>
          <Link 
            to="/doctors" 
            className="inline-block bg-primary text-white px-8 py-3 rounded-lg font-medium hover:bg-primary-800 transition-all"
          >
            BOOK NOW
          </Link>
        </div>
      </div>

      {/* Mental Health Event Section */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h1 className="text-4xl font-bold mb-6 text-gray-900">Mental Health Event Invitation</h1>
        <p className="text-lg mb-8 text-gray-600">
          Join Our Mental Health Awareness Event We invite you to participate in our
          upcoming mental health event. This event is designed to provide valuable
          insights, resources, and support to help you manage your mental health
          effectively. Connect with experts and peers in a supportive environment.
        </p>
        <a 
          href="https://www.earlydetectionresearch.com/" 
          target="_blank" 
          rel="noopener noreferrer" 
          className="inline-block bg-primary text-white px-8 py-3 rounded-full font-medium hover:bg-primary-800 transition-all"
        >
          Event Link
        </a>
        
        {/* Quick Links Grid */}
        <div className="grid grid-cols-2 gap-6 max-w-2xl mx-auto mt-12">
          <Link 
            to="/diagnosis"
            className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg hover:border-primary-100 transition-all group"
          >
            <h3 className="text-xl font-semibold text-gray-900 group-hover:text-primary transition-colors">Diagnosis</h3>
            <p className="mt-2 text-gray-600">Learn about our diagnostic process and early detection methods.</p>
          </Link>
          <Link 
            to="/success-stories"
            className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg hover:border-primary-100 transition-all group"
          >
            <h3 className="text-xl font-semibold text-gray-900 group-hover:text-primary transition-colors">Success Stories</h3>
            <p className="mt-2 text-gray-600">Read inspiring stories of recovery and hope from our patients.</p>
          </Link>
          <Link 
            to="/how-to-deal"
            className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg hover:border-primary-100 transition-all group"
          >
            <h3 className="text-xl font-semibold text-gray-900 group-hover:text-primary transition-colors">How to Deal</h3>
            <p className="mt-2 text-gray-600">Get guidance on coping with diagnosis and treatment.</p>
          </Link>
          <Link 
            to="/risk-factors"
            className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg hover:border-primary-100 transition-all group"
          >
            <h3 className="text-xl font-semibold text-gray-900 group-hover:text-primary transition-colors">Risk Factors</h3>
            <p className="mt-2 text-gray-600">Understand potential risk factors and prevention methods.</p>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ForYou; 