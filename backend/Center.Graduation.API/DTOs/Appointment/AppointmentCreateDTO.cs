namespace Center.Graduation.API.DTOs.Appointment
{
    public class AppointmentCreateDTO
    {
        public int Id { get; set; }
        public string DoctorId { get; set; }
        //public DateTime AppointmentDate { get; set; } // includes date + time
        public DayOfWeek DayOfWeek { get; set; }
        public TimeOnly Time { get; set; }
    }
}
