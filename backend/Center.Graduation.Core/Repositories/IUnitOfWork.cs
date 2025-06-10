using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Core.Repositories
{
    public interface IUnitOfWork
    {
        public IDepartmentRepository departmentRepository { get; }
        public IAppointmentRepository appointmentRepository { get; }
        public IDoctorWorkTimeRepository doctorWorkTimeRepository { get; }
    }
}
