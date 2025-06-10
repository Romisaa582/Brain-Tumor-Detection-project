using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Core.Entities
{
    public class DoctorWorkTime
    {
        public int Id { get; set; }

        public string DoctorId { get; set; }
        public ApplicationUser Doctor { get; set; }
        public DayOfWeek Day { get; set; }   
        public TimeOnly StartTime { get; set; }
        public TimeOnly EndTime { get; set; }
    }
}