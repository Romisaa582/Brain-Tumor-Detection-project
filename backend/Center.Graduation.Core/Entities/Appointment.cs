using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Core.Entities
{
    public class Appointment
    {
        public int Id { get; set; }
        public string PatientId { get; set; }    //Fk
        public ApplicationUser Patient { get; set; }

        public string DoctorId { get; set; }
        public ApplicationUser Doctor { get; set; }
        public DayOfWeek DayOfWeek { get; set; }
        public TimeOnly Time { get; set; }
    }
}
