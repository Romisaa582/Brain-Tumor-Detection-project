import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';

const timeSlots = [
  '8:00 am', '9:00 am', '10:00 am', '11:00 am', 
  '1:00 pm', '2:00 pm', '3:00 pm', '4:00 pm'
];

// دالة لتوليد الأيام والتواريخ ديناميكياً
const generateDays = () => {
  const days = [];
  const today = new Date();

  for (let i = 0; i < 7; i++) {
    const date = new Date(today);
    date.setDate(today.getDate() + i);

    days.push({
      day: date.toLocaleDateString('en-US', { weekday: 'short' }).toUpperCase(),
      date: date.getDate().toString(),
      fullName: date.toLocaleDateString('en-US', { weekday: 'long' }),
      fullDate: date.toISOString().split('T')[0] // YYYY-MM-DD format
    });
  }

  return days;
};

// دالة لتوليد كل المواعيد من 08:00 إلى 19:30
const generateTimeSlots = () => {
  const slots = [];
  let hour = 8;
  let minute = 0;

  while (hour < 20) { // نستمر حتى 19:30
    const formattedHour = hour.toString().padStart(2, '0');
    const formattedMinute = minute.toString().padStart(2, '0');
    slots.push(`${formattedHour}:${formattedMinute}`);

    // زيادة 30 دقيقة
    if (minute === 0) {
      minute = 30;
    } else {
      minute = 0;
      hour++;
    }
  }
  return slots;
};

// مكون جديد لعرض تفاصيل الموعد
const AppointmentDetails = ({ appointment }) => {
  const [timeLeft, setTimeLeft] = useState('');

  useEffect(() => {
    const calculateTimeLeft = () => {
      const appointmentDate = new Date();
      const [hours, minutes] = appointment.time.split(':');
      appointmentDate.setHours(parseInt(hours), parseInt(minutes), 0, 0);

      // تحديد التاريخ الصحيح بناءً على اليوم
      const today = new Date();
      const daysOfWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
      const dayDiff = daysOfWeek.indexOf(appointment.dayOfWeek) - today.getDay();
      appointmentDate.setDate(today.getDate() + (dayDiff >= 0 ? dayDiff : dayDiff + 7));

      const now = new Date();
      const difference = appointmentDate - now;

      if (difference > 0) {
        const days = Math.floor(difference / (1000 * 60 * 60 * 24));
        const hours = Math.floor((difference % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((difference % (1000 * 60 * 60)) / (1000 * 60));
        
        setTimeLeft(`${days} days, ${hours} hours, ${minutes} minutes`);
      } else {
        setTimeLeft('Appointment time has passed');
      }
    };

    calculateTimeLeft();
    const timer = setInterval(calculateTimeLeft, 60000); // تحديث كل دقيقة

    return () => clearInterval(timer);
  }, [appointment]);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mt-8">
      <h3 className="text-xl font-semibold mb-4 text-indigo-600">Your Upcoming Appointment</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span className="text-gray-700">Day: <span className="font-semibold">{appointment.dayOfWeek}</span></span>
          </div>
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-gray-700">Time: <span className="font-semibold">{appointment.time}</span></span>
          </div>
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
            <span className="text-gray-700">Doctor: <span className="font-semibold">Dr. {appointment.doctorName}</span></span>
          </div>
        </div>
        <div className="bg-indigo-50 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-indigo-600 mb-2">Time Remaining</h4>
          <div className="text-indigo-800">
            {timeLeft}
          </div>
        </div>
      </div>
    </div>
  );
};

const Appointment = () => {
  const { docId } = useParams();
  const navigate = useNavigate();
  const [doctor, setDoctor] = useState(null);
  const [workTimes, setWorkTimes] = useState([]);
  const [bookedAppointments, setBookedAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedDay, setSelectedDay] = useState(null);
  const [selectedTime, setSelectedTime] = useState(null);
  const [relatedDoctors, setRelatedDoctors] = useState([]);
  const [bookedSlots, setBookedSlots] = useState([]);
  const [userAppointment, setUserAppointment] = useState(null);
  const [bookingStatus, setBookingStatus] = useState({ loading: false, error: null, success: false });

  // استخدام الدالة لتوليد الأيام
  const [days, setDays] = useState(generateDays());

  const timeSlots = generateTimeSlots();

  // تحديث الأيام كل يوم جديد
  useEffect(() => {
    setDays(generateDays());
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = localStorage.getItem('token');
        
        // جلب بيانات الدكتور
        const doctorResponse = await axios.get(
          `https://tumortraker12.runasp.net/api/Account/GetUserById?userId=${docId}`,
          {
            headers: { Authorization: `Bearer ${token}` }
          }
        );
        setDoctor(doctorResponse.data);

        // جلب مواعيد عمل الدكتور
        const workTimesResponse = await axios.get(
          `https://tumortraker12.runasp.net/api/DoctorWorkTime?DoctorId=${docId}`,
          {
            headers: { Authorization: `Bearer ${token}` }
          }
        );
        setWorkTimes(workTimesResponse.data);

        // جلب المواعيد المحجوزة
        const bookedResponse = await axios.get(
          `https://tumortraker12.runasp.net/api/Appointment/GetDoctorAppointment?DoctorId=${docId}`,
          {
            headers: { Authorization: `Bearer ${token}` }
          }
        );
        setBookedAppointments(bookedResponse.data);
        setLoading(false);
      } catch (err) {
        console.error('Error:', err);
        setError('حدث خطأ أثناء تحميل البيانات');
        setLoading(false);
      }
    };

    fetchData();
  }, [docId]);

  useEffect(() => {
    const fetchUserAppointment = async () => {
    try {
      const token = localStorage.getItem('token');
        // افترض أن هناك API يجلب موعد المريض الحالي
        const response = await axios.get(
          `https://tumortraker12.runasp.net/api/Appointment/GetPatientAppointment`,
          {
            headers: { Authorization: `Bearer ${token}` }
          }
        );
        if (response.data) {
          setUserAppointment(response.data);
        }
    } catch (err) {
        console.error('Error fetching user appointment:', err);
      }
    };
    image.png
    fetchUserAppointment();
  }, []);

  // دالة للتحقق ما إذا كان الوقت متاح في مواعيد عمل الدكتور
  const isTimeSlotAvailable = (day, time) => {
    // التحقق من أن الوقت ضمن ساعات عمل الدكتور
    const workTime = workTimes.find(wt => wt.day === day.fullName);
    if (!workTime) return false;

    const timeStr = time.replace(/^(\d{1}):/, '0$1:'); // تنسيق الوقت
    if (timeStr < workTime.startTime || timeStr > workTime.endTime) return false;

    // التحقق من أن الموعد غير محجوز
    const isBooked = bookedAppointments.some(
      appointment => 
        appointment.dayOfWeek === day.fullName && 
        appointment.time === timeStr
    );

    return !isBooked;
  };

  // دالة لتحديث المواعيد المحجوزة
  const updateBookedSlots = (selectedDay) => {
    // هنا يمكنك جلب المواعيد المحجوزة من API
    // مثال مؤقت:
    if (selectedDay.day === 'MON') {
      setBookedSlots(['08:30', '09:00', '15:30']);
    } else if (selectedDay.day === 'WED') {
      setBookedSlots(['10:00', '14:30']);
    } else {
      setBookedSlots([]);
    }
  };

  // تحديث عند اختيار يوم
  const handleDaySelection = (day) => {
    setSelectedDay(day);
    setSelectedTime(null);
    updateBookedSlots(day);
  };

  // دالة حجز الموعد
  const bookAppointment = async () => {
    // التحقق من وجود البيانات الأساسية
    if (!selectedDay || !selectedTime) {
      setBookingStatus({ 
        loading: false, 
        error: 'الرجاء اختيار اليوم والوقت المناسب', 
        success: false 
      });
      return;
    }

    // التحقق من صحة التاريخ (لا يمكن حجز موعد في يوم سابق)
    const today = new Date();
    const selectedDate = new Date(selectedDay.fullDate);
    selectedDate.setHours(parseInt(selectedTime.split(':')[0]), parseInt(selectedTime.split(':')[1]));
    
    if (selectedDate < today) {
      setBookingStatus({
        loading: false,
        error: 'لا يمكن حجز موعد في وقت سابق',
        success: false
      });
      return;
    }

    // التحقق من وجود معرف الدكتور
    if (!docId) {
      setBookingStatus({
        loading: false,
        error: 'بيانات الدكتور غير صحيحة',
        success: false
      });
      return;
    }

    // التحقق من وجود التوكن
    const token = localStorage.getItem('token');
    if (!token) {
      setBookingStatus({
        loading: false,
        error: 'يرجى تسجيل الدخول أولاً',
        success: false
      });
      navigate('/login');
      return;
    }

    // التحقق من أن الموعد ضمن ساعات عمل الدكتور
    const workTime = workTimes.find(wt => wt.day === selectedDay.fullName);
    if (!workTime) {
      setBookingStatus({
        loading: false,
        error: 'الدكتور غير متاح في هذا اليوم',
        success: false
      });
      return;
    }

    const timeStr = selectedTime.replace(/^(\d{1}):/, '0$1:');
    if (timeStr < workTime.startTime || timeStr > workTime.endTime) {
      setBookingStatus({
        loading: false,
        error: 'الوقت المختار خارج ساعات عمل الدكتور',
        success: false
      });
      return;
    }

    // التحقق من أن الموعد غير محجوز
    const isBooked = bookedAppointments.some(
      appointment => 
        appointment.dayOfWeek === selectedDay.fullName && 
        appointment.time === timeStr
    );

    if (isBooked) {
      setBookingStatus({
        loading: false,
        error: 'هذا الموعد محجوز بالفعل. الرجاء اختيار موعد آخر',
        success: false
      });
      return;
    }

    // التأكد من أن المستخدم يريد حجز الموعد
    if (!window.confirm(`هل أنت متأكد من حجز موعد مع الدكتور ${doctor?.firstName} ${doctor?.lastName}؟\n\nاليوم: ${selectedDay.fullName}\nالوقت: ${selectedTime}`)) {
      return;
    }

    setBookingStatus({ loading: true, error: null, success: false });
    
    try {
      const appointmentData = {
        id: 0,
        doctorId: docId,
        dayOfWeek: selectedDay.fullName,
        time: timeStr // استخدام الوقت المنسق
      };

      const response = await axios.post(
        'https://tumortraker12.runasp.net/api/Appointment',
        appointmentData,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.data) {
        setBookingStatus({ loading: false, error: null, success: true });
        
        try {
          // تحديث البيانات
          const [updatedBookings, updatedUserAppointment] = await Promise.all([
            axios.get(
              `https://tumortraker12.runasp.net/api/Appointment/GetDoctorAppointment?DoctorId=${docId}`,
              { headers: { Authorization: `Bearer ${token}` } }
            ),
            axios.get(
              `https://tumortraker12.runasp.net/api/Appointment/GetPatientAppointment`,
              { headers: { Authorization: `Bearer ${token}` } }
            )
          ]);

          setBookedAppointments(updatedBookings.data);
          setUserAppointment(updatedUserAppointment.data);
        } catch (updateErr) {
          console.error('Error updating appointments:', updateErr);
        }
        
        // إعادة تعيين الاختيارات
        setSelectedDay(null);
        setSelectedTime(null);

        // عرض رسالة النجاح
        window.alert('تم حجز الموعد بنجاح!');
      }
    } catch (err) {
      console.error('Booking error:', err);
      let errorMessage = 'فشل في حجز الموعد. الرجاء المحاولة مرة أخرى.';
      
      if (err.response) {
        // الحصول على رسالة الخطأ من الخادم إذا كانت متوفرة
        errorMessage = err.response.data?.message || errorMessage;
        
        switch (err.response.status) {
          case 400:
            errorMessage = 'بيانات الحجز غير صحيحة. الرجاء التحقق من المعلومات المدخلة.';
            break;
          case 401:
            errorMessage = 'انتهت صلاحية الجلسة. يرجى تسجيل الدخول مرة أخرى.';
            localStorage.removeItem('token');
            navigate('/login');
            break;
          case 403:
            errorMessage = 'غير مصرح لك بحجز هذا الموعد.';
            break;
          case 409:
            errorMessage = 'هذا الموعد محجوز بالفعل. الرجاء اختيار موعد آخر.';
            // تحديث المواعيد المحجوزة
            if (token) {
              try {
                const updatedBookings = await axios.get(
                  `https://tumortraker12.runasp.net/api/Appointment/GetDoctorAppointment?DoctorId=${docId}`,
                  { headers: { Authorization: `Bearer ${token}` } }
                );
                setBookedAppointments(updatedBookings.data);
              } catch (updateErr) {
                console.error('Error updating booked appointments:', updateErr);
              }
            }
            break;
          case 500:
            errorMessage = 'حدث خطأ في النظام. الرجاء المحاولة مرة أخرى لاحقاً.';
            break;
        }
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setBookingStatus({ 
        loading: false, 
        error: errorMessage, 
        success: false 
      });
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center p-4">
        <div className="text-red-600">{error}</div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Doctor Info Card */}
      <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
        <div className="flex gap-10">
          {/* Doctor Image */}
          <div className="w-48 h-48 rounded-xl overflow-hidden flex-shrink-0 shadow-md">
            <img
              src={doctor?.photoURL || 'default-doctor-image.png'}
              alt={`Dr. ${doctor?.firstName}`}
              className="w-full h-full object-cover"
            />
          </div>
          
          {/* Doctor Info */}
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-4">
              <h2 className="text-3xl font-bold text-gray-800">
                Dr. {doctor?.firstName} {doctor?.lastName}
              </h2>
              <span className="text-blue-600 bg-blue-50 p-1.5 rounded-full">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              </span>
            </div>

            {/* Specialization Badge */}
            <div className="mb-6">
              <div className="inline-flex items-center px-4 py-2 rounded-full bg-blue-50 border border-blue-100">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
                <span className="text-xl text-blue-700 font-semibold">
                  {doctor?.departmentId === 1 ? 'Psychologist' : 'Oncologist'}
                </span>
              </div>
            </div>

            {/* Doctor Details - 2 columns */}
            <div className="grid grid-cols-2 gap-6 mb-6">
              {doctor?.email && (
                <div className="flex items-center gap-3 text-gray-600">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" />
                    <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" />
                    </svg>
                  <span className="text-lg">{doctor.email}</span>
                </div>
              )}

              {doctor?.websiteURL && (
                <div className="flex items-center gap-3 text-gray-600">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.083 9h1.946c.089-1.546.383-2.97.837-4.118A6.004 6.004 0 004.083 9zM10 2a8 8 0 100 16 8 8 0 000-16zm0 2c-.076 0-.232.032-.465.262-.238.234-.497.623-.737 1.182-.389.907-.673 2.142-.766 3.556h3.936c-.093-1.414-.377-2.649-.766-3.556-.24-.56-.5-.948-.737-1.182C10.232 4.032 10.076 4 10 4zm3.971 5c-.089-1.546-.383-2.97-.837-4.118A6.004 6.004 0 0115.917 9h-1.946zm-2.003 2H8.032c.093 1.414.377 2.649.766 3.556.24.56.5.948.737 1.182.233.23.389.262.465.262.076 0 .232-.032.465-.262.238-.234.498-.623.737-1.182.389-.907.673-2.142.766-3.556zm1.166 4.118c.454-1.147.748-2.572.837-4.118h1.946a6.004 6.004 0 01-2.783 4.118zm-6.268 0C6.412 13.97 6.118 12.546 6.03 11H4.083a6.004 6.004 0 002.783 4.118z" clipRule="evenodd" />
                  </svg>
                  <a href={doctor.websiteURL} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline text-lg">
                    Visit Website
                  </a>
                </div>
              )}
            </div>

            {/* About Section */}
            <div className="mb-6">
              <h3 className="text-xl font-semibold mb-3">About</h3>
              <p className="text-gray-600 leading-relaxed text-lg">
                {doctor?.about || 
                 `Experienced ${doctor?.departmentId === 1 ? 'Psychologist' : 'Oncologist'} committed to delivering comprehensive medical care, focusing on ${doctor?.departmentId === 1 ? 'mental health and psychological well-being' : 'cancer diagnosis, treatment, and patient care'}. Providing personalized care tailored to each patient's needs.`}
              </p>
            </div>

            {/* Appointment Fee and Chat Button */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3 bg-blue-50 p-4 rounded-xl">
                <span className="text-xl font-semibold text-gray-800">Appointment Fee:</span>
                <span className="text-2xl font-bold text-blue-600">${doctor?.appointmentFee || '50'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* عرض موعد المريض الحالي إذا وجد */}
      {userAppointment && (
        <AppointmentDetails appointment={userAppointment} />
      )}

      {/* Appointments Section */}
      <div className="mt-8">
        <h3 className="text-xl font-semibold mb-4">Select Day</h3>
        <div className="grid grid-cols-7 gap-4 mb-8">
          {days.map((day) => {
            const hasWorkTime = workTimes.some(wt => wt.day === day.fullName);
            return (
                <button
                  key={day.day}
                onClick={() => setSelectedDay(day)}
                className={`p-4 rounded-xl flex flex-col items-center justify-center ${
                  selectedDay?.day === day.day
                    ? 'bg-indigo-600 text-white'
                    : hasWorkTime
                    ? 'bg-green-50 hover:bg-green-100'
                    : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                }`}
                disabled={!hasWorkTime}
              >
                <span className="text-lg font-semibold mb-1">{day.day}</span>
                <span className="text-sm">{day.date}</span>
                {hasWorkTime && (
                  <span className="text-xs mt-1 text-green-600">Available</span>
                )}
                </button>
            );
          })}
            </div>

            {/* Time Slots - Now always visible */}
            <div className="mt-8">
              <h3 className="text-xl font-semibold mb-4">Available Time Slots</h3>
              <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-3">
                {generateTimeSlots().map((time) => {
                  const isAvailable = selectedDay ? isTimeSlotAvailable(selectedDay, time) : false;
                  const isBooked = selectedDay ? bookedAppointments.some(
                    app => app.dayOfWeek === selectedDay.fullName && app.time === time
                  ) : false;
                  const isSelected = selectedTime === time;
                  
                  return (
                    <button
                      key={time}
                      onClick={() => selectedDay && setSelectedTime(time)}
                      className={`p-3 rounded-lg text-center ${
                        isSelected
                          ? 'bg-indigo-600 text-white'
                          : !selectedDay
                          ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                          : isAvailable
                          ? 'bg-green-50 hover:bg-green-100 text-green-700'
                          : isBooked
                          ? 'bg-red-50 text-red-400 cursor-not-allowed'
                          : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      }`}
                      disabled={!selectedDay || !isAvailable || isBooked}
                    >
                      {time}
                      {isBooked && (
                        <span className="block text-xs mt-1">Booked</span>
                      )}
                      {!selectedDay && (
                        <span className="block text-xs mt-1">Select a day</span>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Booking Button and Status */}
              {selectedTime && (
                <div className="mt-8">
                  <button
                    onClick={bookAppointment}
                    disabled={bookingStatus.loading}
                    className={`px-6 py-3 rounded-lg text-white font-semibold ${
                      bookingStatus.loading
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-indigo-600 hover:bg-indigo-700'
                    }`}
                  >
                    {bookingStatus.loading ? 'جاري الحجز...' : 'تأكيد الحجز'}
                  </button>

                  {bookingStatus.error && (
                    <div className="mt-4 p-3 bg-red-50 text-red-600 rounded-lg">
                      {bookingStatus.error}
                    </div>
                  )}

                  {bookingStatus.success && (
                    <div className="mt-4 p-3 bg-green-50 text-green-600 rounded-lg">
                      تم حجز الموعد بنجاح!
                    </div>
                  )}

                  <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-semibold text-blue-800 mb-2">تفاصيل الحجز:</h4>
                    <p className="text-blue-700">
                      اليوم: {selectedDay.fullName}<br />
                      الوقت: {selectedTime}<br />
                      الدكتور: Dr. {doctor?.firstName} {doctor?.lastName}
                    </p>
                  </div>
                </div>
              )}
            </div>
      </div>

            {/* Related Doctors */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Related Doctors</h3>
        <p className="text-sm text-gray-600 mb-4">
          Similar doctors from our trusted list of medical practitioners.
        </p>
        
        <div className="grid grid-cols-5 gap-4">
          {[1,2,3,4,5].map((_, index) => (
            <div key={index} className="text-center">
              <div className="w-full aspect-square rounded-lg bg-gray-100 mb-2">
                <img
                  src={doctor?.photoURL || 'default-doctor-image.png'}
                  alt="Doctor"
                  className="w-full h-full object-cover rounded-lg"
                    />
                  </div>
                    <p className="text-xs text-green-600 mb-1">Available</p>
              <h4 className="text-sm font-medium">Dr. Richard James</h4>
              <p className="text-xs text-gray-600">General physician</p>
                </div>
              ))}
            </div>
          </div>
    </div>
  );
};

export default Appointment;