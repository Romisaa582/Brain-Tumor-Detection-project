using Center.Graduation.Core.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Core.Repositories
{
    public interface IAppointmentRepository
    {
        public Task<IEnumerable<Appointment>> GetAllAppointmentOfDoctor(string Id);
        public Task<IEnumerable<Appointment>> GetAllAppointmentOfPatient(string Id);
        public Task<IEnumerable<Appointment>> GatAllPatientAppointmentWithDoctor(string DoctorId , string PatientId);
        public Task<Appointment> GetByIdAsync(int Id);
        public Task<bool> CheckAppointMent(string DoctorId, DayOfWeek dayOfWeek, TimeOnly Time);
        public Task<int> AddAsync(Appointment appointment);
        public Task<int> RemoveAsync(Appointment appointment);
    }
}