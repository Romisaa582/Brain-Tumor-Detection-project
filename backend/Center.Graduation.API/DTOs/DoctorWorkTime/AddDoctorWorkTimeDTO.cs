using Center.Graduation.Core.Entities;
using System.ComponentModel.DataAnnotations;

namespace Center.Graduation.API.DTOs.DoctorWorkTime
{
    public class AddDoctorWorkTimeDTO
    {
        public int Id { get; set; }

        [Required(ErrorMessage ="Start Time is required")]
        public DayOfWeek Day { get; set; }          // Enum: Sunday to Saturday

        [Required(ErrorMessage ="Start Time is required")]
        public TimeOnly StartTime { get; set; }
               
        [Required(ErrorMessage ="End Time is required")]
        public TimeOnly EndTime { get; set; }
    }
}
