import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const About = () => {
  const teamMembers = [
    {
      name: 'Dr. Sarah Johnson',
      role: 'Lead Oncologist',
      image: 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?q=80&w=200&h=200&auto=format&fit=crop',
      description: 'Specializes in early cancer detection and treatment planning.'
    },
    {
      name: 'Dr. Michael Chen',
      role: 'AI Research Lead',
      image: 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?q=80&w=200&h=200&auto=format&fit=crop',
      description: 'Expert in machine learning and medical imaging analysis.'
    },
    {
      name: 'Dr. Emily Rodriguez',
      role: 'Clinical Director',
      image: 'https://images.unsplash.com/photo-1594824476967-48c8b964273f?q=80&w=200&h=200&auto=format&fit=crop',
      description: 'Oversees patient care and treatment protocols.'
    }
  ];

  const technologies = [
    {
      name: 'AI-Powered Detection',
      description: 'Advanced machine learning algorithms for accurate tumor detection',
      icon: (
        <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      )
    },
    {
      name: 'Real-time Analysis',
      description: 'Instant processing of medical imaging for quick results',
      icon: (
        <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      )
    },
    {
      name: 'Secure Platform',
      description: 'HIPAA-compliant security for patient data protection',
      icon: (
        <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
      )
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-gradient-to-r from-indigo-600 to-blue-500">
        <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-transparent z-0" />
        <div className="max-w-7xl mx-auto relative z-10">
          <div className="relative pb-8 sm:pb-16 md:pb-20 lg:pb-28 xl:pb-32">
            <main className="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8 xl:mt-28">
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className="sm:text-center lg:text-left"
              >
                <h1 className="text-4xl tracking-tight font-extrabold text-white sm:text-5xl md:text-6xl">
                  <span className="block">Revolutionizing</span>
                  <span className="block text-blue-200">Cancer Detection</span>
                </h1>
                <p className="mt-3 text-base text-gray-100 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
                  We're combining advanced AI technology with medical expertise to provide early cancer detection and improve patient outcomes.
                </p>
                <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
                  <div className="rounded-md shadow">
                    <Link
                      to="/doctors"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-full text-indigo-600 bg-white hover:bg-gray-50 md:py-4 md:text-lg md:px-10 transition-all duration-300 hover:shadow-lg"
                    >
                      Find a Doctor
                    </Link>
                  </div>
                  <div className="mt-3 sm:mt-0 sm:ml-3">
                    <Link
                      to="/contact"
                      className="w-full flex items-center justify-center px-8 py-3 border-2 border-white text-base font-medium rounded-full text-white hover:bg-white hover:text-indigo-600 md:py-4 md:text-lg md:px-10 transition-all duration-300"
                    >
                      Contact Us
                    </Link>
                  </div>
                </div>
              </motion.div>
            </main>
          </div>
        </div>
      </div>

      {/* Mission Section */}
      <div className="py-16 bg-white overflow-hidden lg:py-24">
        <div className="relative max-w-xl mx-auto px-4 sm:px-6 lg:px-8 lg:max-w-7xl">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="relative"
          >
            <h2 className="text-center text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              Our Mission
            </h2>
            <p className="mt-4 max-w-3xl mx-auto text-center text-xl text-gray-500">
              To leverage cutting-edge AI technology in making cancer detection more accessible, accurate, and timely, ultimately saving lives through early intervention.
            </p>
          </motion.div>

          {/* Technologies Section */}
          <div className="mt-16">
            <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
              {technologies.map((tech, index) => (
                <motion.div 
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.2 }}
                  viewport={{ once: true }}
                  className="pt-6"
                >
                  <div className="flow-root bg-gradient-to-br from-indigo-600 to-blue-500 rounded-2xl px-6 pb-8 h-full text-white hover:shadow-xl transition-all duration-300">
                    <div className="-mt-6">
                      <div>
                        <span className="inline-flex items-center justify-center p-3 bg-white rounded-xl shadow-lg ring-4 ring-white">
                          {tech.icon}
                        </span>
                      </div>
                      <h3 className="mt-8 text-xl font-semibold tracking-tight">{tech.name}</h3>
                      <p className="mt-5 text-base text-blue-100">{tech.description}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Team Section */}
      <div className="bg-gray-50">
        <div className="mx-auto py-12 px-4 max-w-7xl sm:px-6 lg:px-8 lg:py-24">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="space-y-12"
          >
            <div className="space-y-5 sm:space-y-4 md:max-w-xl lg:max-w-3xl xl:max-w-none">
              <h2 className="text-3xl font-extrabold tracking-tight sm:text-4xl">Our Team</h2>
              <p className="text-xl text-gray-500">
                Meet our team of experienced medical professionals and technology experts dedicated to improving cancer detection and patient care.
              </p>
            </div>
            <div className="grid grid-cols-1 gap-12 sm:grid-cols-2 lg:grid-cols-3">
              {teamMembers.map((member, index) => (
                <motion.div 
                  key={index}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.2 }}
                  viewport={{ once: true }}
                  className="space-y-4 group"
                >
                  <div className="aspect-w-3 aspect-h-3 rounded-2xl overflow-hidden transform transition-all duration-300 group-hover:shadow-2xl">
                    <img
                      className="object-cover w-full h-full transform transition-transform duration-300 group-hover:scale-110"
                      src={member.image}
                      alt={member.name}
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="text-lg leading-6 font-medium space-y-1">
                      <h3 className="text-xl font-bold">{member.name}</h3>
                      <p className="text-indigo-600 font-semibold">{member.role}</p>
                    </div>
                    <div className="text-base text-gray-500">
                      <p>{member.description}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-gradient-to-r from-indigo-600 to-blue-500">
        <div className="max-w-7xl mx-auto py-12 px-4 sm:py-16 sm:px-6 lg:px-8 lg:py-20">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto text-center"
          >
            <h2 className="text-3xl font-extrabold text-white sm:text-4xl">
              Trusted by Healthcare Providers Worldwide
            </h2>
            <p className="mt-3 text-xl text-blue-100 sm:mt-4">
              Our platform is making a real difference in cancer detection and patient care.
            </p>
          </motion.div>
          <dl className="mt-10 text-center sm:max-w-3xl sm:mx-auto sm:grid sm:grid-cols-3 sm:gap-8">
            {[
              { label: 'Success Rate', value: '95%' },
              { label: 'Patients Helped', value: '10k+' },
              { label: 'Partner Hospitals', value: '100+' }
            ].map((stat, index) => (
              <motion.div 
                key={index}
                initial={{ opacity: 0, scale: 0.5 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="flex flex-col"
              >
                <dt className="order-2 mt-2 text-lg leading-6 font-medium text-blue-100">
                  {stat.label}
                </dt>
                <dd className="order-1 text-5xl font-extrabold text-white">{stat.value}</dd>
              </motion.div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  );
};

export default About;