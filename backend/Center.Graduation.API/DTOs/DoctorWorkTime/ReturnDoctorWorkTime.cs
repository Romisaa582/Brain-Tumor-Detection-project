namespace Center.Graduation.API.DTOs.DoctorWorkTime
{
    public class ReturnDoctorWorkTime
    {
        public int Id { get; set; }
        public string DoctorId { get; set; }
        public string DoctorName { get; set; }
        public string Day { get; set; }
        public TimeOnly StartTime { get; set; }
        public TimeOnly EndTime { get; set; }


    }
}
