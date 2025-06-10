using Center.Graduation.Core.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Core.Repositories
{
    public interface IDoctorWorkTimeRepository
    {
        public Task<IEnumerable<DoctorWorkTime>> GetAllDoctorWorkTime(string Id);
        public Task<DoctorWorkTime> GetByIdAsync(int Id);

        public Task<DoctorWorkTime> CheckDoctorWorkTime(string DoctorId, DayOfWeek dayOfWeek, TimeOnly Time);
        public Task<IEnumerable<DoctorWorkTime>> GetAll(int Id);
        public Task<int> AddAsync(DoctorWorkTime doctorWorkTime);
        public Task<int> RemoveAsync(DoctorWorkTime doctorWorkTime);
    }
}
