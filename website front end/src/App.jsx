import React, { useState, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Home from './pages/Home'
import Doctor from './pages/Doctor'
import Login from './pages/Login'
import About from './pages/About'
import MyProfile from './pages/MyProfile'
import MyAppointment from './pages/MyAppointment'
import Appointment from './pages/Appointment'
import Contact from './pages/Contact'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import { useAuth } from './context/AuthContext'
import Register from './pages/Register'
import AiTool from './pages/AiTool'
import MyPatients from './pages/MyPatients'
import Dashboard from './pages/Dashboard'
import ProtectedRoute from './components/ProtectedRoute'
import DoctorSchedule from './pages/DoctorSchedule'
import VideoCall from './components/VideoCall'
import DoctorAppointments from './pages/DoctorAppointments'

const App = () => {
  const { user } = useAuth();

  return (
    <div className='mx-4 sm:mx-[10%]'>
      <Navbar />
      <Routes>
        <Route path='/login' element={
          user ? (user.role === 'Patient' ? <Navigate to="/" /> : <Navigate to="/my-profile" />) : <Login />
        } />
        <Route path='/register' element={
          user ? (user.role === 'Patient' ? <Navigate to="/" /> : <Navigate to="/my-profile" />) : <Register />
        } />
        <Route path='/dashboard' element={
          <ProtectedRoute allowedRoles={['Doctor']}>
            <Dashboard />
          </ProtectedRoute>
        } />
        <Route path='/schedule' element={
          <ProtectedRoute allowedRoles={['Doctor']}>
            <DoctorSchedule />
          </ProtectedRoute>
        } />
        <Route path='/doctors' element={
          <ProtectedRoute allowedRoles={['Patient']}>
            <Doctor />
          </ProtectedRoute>
        } />
        <Route path='/doctors/:speciality' element={
          <ProtectedRoute allowedRoles={['Patient']}>
            <Doctor />
          </ProtectedRoute>
        } />
        <Route path='/my-patients' element={
          <ProtectedRoute allowedRoles={['Doctor']}>
            <MyPatients />
          </ProtectedRoute>
        } />
        <Route path='/ai-tool' element={
          <ProtectedRoute allowedRoles={['Patient']}>
            <AiTool />
          </ProtectedRoute>
        } />
        <Route path='/about' element={<About />} />
        <Route path='/contact' element={<Contact />} />
        <Route path='/my-profile' element={
          <ProtectedRoute>
            <MyProfile />
          </ProtectedRoute>
        } />
        <Route path='/my-appointments' element={
          <ProtectedRoute>
            <MyAppointment />
          </ProtectedRoute>
        } />
        <Route path='/appointment' element={
          <ProtectedRoute>
            <Appointment />
          </ProtectedRoute>
        } />
        <Route path='/appointment/:docId' element={
          <ProtectedRoute>
            <Appointment />
          </ProtectedRoute>
        } />
        <Route path='/video-call' element={
          <ProtectedRoute>
            <VideoCall />
          </ProtectedRoute>
        } />
        <Route path='/doctor-appointments' element={
          <ProtectedRoute allowedRoles={['Doctor']}>
            <DoctorAppointments />
          </ProtectedRoute>
        } />
        <Route path='/' element={
          user ? <Home /> : <Navigate to="/login" />
        } />
      </Routes>
      <Footer />
    </div>
  )
}

export default App