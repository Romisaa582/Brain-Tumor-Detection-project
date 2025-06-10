using Center.Graduation.Core.Entities;

namespace Center.Graduation.API.DTOs.Appointment
{
    public class GetAppointment
    {
        public int Id { get; set; }
        public string DoctorId { get; set; }
        public string DoctorName { get; set; }
        public string PatientId { get; set; }
        public string PatientName { get; set; }
        public DayOfWeek DayOfWeek { get; set; }
        public TimeOnly Time { get; set; }
    }
}
