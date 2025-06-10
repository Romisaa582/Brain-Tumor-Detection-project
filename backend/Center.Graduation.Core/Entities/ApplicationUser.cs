using Center.Graduation.Core.Repositories;
using Microsoft.AspNetCore.Identity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Core.Entities
{
    public class ApplicationUser:IdentityUser     // userName     Email    Password    PhoneNumber 
    {
        public string FirstName { get; set; }
        public string LastName { get; set; }

        public string? PhotoURL { get; set; }
        public int Age { get; set; }

        public string? WebsiteURL { get; set; }
        public string Type { get; set; }
        public int? DepartmentId { get; set; }
        public Department? Department { get; set; }
        public List<DoctorWorkTime>? doctorWorkTimes{ get; set; }

        //المرض و الاعراض 

    }
}
