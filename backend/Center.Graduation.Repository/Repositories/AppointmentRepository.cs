using Center.Graduation.Core.Entities;
using Center.Graduation.Core.Repositories;
using Center.Graduation.Repository.Contexts;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Repository.Repositories
{
    public class AppointmentRepository : IAppointmentRepository
    {
        private readonly ApplicationDbContext _context;

        public AppointmentRepository(ApplicationDbContext context)
        {
            _context = context;
        }

        public async Task<int> AddAsync(Appointment appointment)
        {
           await _context.Appointments.AddAsync(appointment);
            return await _context.SaveChangesAsync();
        }

        public async Task<bool> CheckAppointMent(string DoctorId , DayOfWeek dayOfWeek , TimeOnly Time) 
        {
            var isTaken = await _context.Appointments.AnyAsync(a =>
               a.DoctorId == DoctorId &&
               a.DayOfWeek == dayOfWeek &&
               a.Time == Time);
            return isTaken; 
        }

        public async Task<IEnumerable<Appointment>> GetAllAppointmentOfDoctor(string Id)
            => await _context.Appointments.
                Include(e=>e.Doctor)
                .Include(e=>e.Patient)
                .Where(e=>e.DoctorId==Id).ToListAsync();

        public async Task<IEnumerable<Appointment>> GetAllAppointmentOfPatient(string Id)
           => await _context.Appointments.
               Include(e => e.Doctor)
               .Include(e => e.Patient)
               .Where(e => e.PatientId == Id).ToListAsync();


        public async Task<IEnumerable<Appointment>> GatAllPatientAppointmentWithDoctor(string DoctorId, string PatientId) 
        {
            var appointments = await _context.Appointments
                .Where(e => e.DoctorId == DoctorId && e.PatientId == PatientId)
                .Include(e => e.Doctor)
                .Include(e => e.Patient)
                .ToListAsync();
            return appointments;
        }


        public async Task<Appointment> GetByIdAsync(int Id)
            => await _context.Appointments.FindAsync(Id);
        public async Task<int> RemoveAsync(Appointment appointment)
        {
             _context.Appointments.Remove(appointment);
            return await _context.SaveChangesAsync();
        }
    }
}
