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
    public class DoctorWorkTimeRepository : IDoctorWorkTimeRepository
    {
        private readonly ApplicationDbContext _context;

        public DoctorWorkTimeRepository(ApplicationDbContext context)
        {
            _context = context;
        }
        public async Task<int> AddAsync(DoctorWorkTime doctorWorkTime)
        {
            await _context.DoctorWorkTimes.AddAsync(doctorWorkTime);
            return await _context.SaveChangesAsync();
        }

        public async Task<DoctorWorkTime> CheckDoctorWorkTime(string DoctorId, DayOfWeek dayOfWeek, TimeOnly Time)
        {
            var workTime = await _context.DoctorWorkTimes
               .FirstOrDefaultAsync(w =>
                   w.DoctorId == DoctorId &&
                   w.Day == dayOfWeek &&
                   Time >= w.StartTime &&
                   Time < w.EndTime); // strictly inside working time
            return workTime; 
        }

        public async Task<IEnumerable<DoctorWorkTime>> GetAll(int Id)
            => await _context.DoctorWorkTimes.ToListAsync();

        public async Task<IEnumerable<DoctorWorkTime>> GetAllDoctorWorkTime(string Id)
            => await _context.DoctorWorkTimes.Where(e=>e.DoctorId== Id).ToListAsync();

        public async Task<DoctorWorkTime> GetByIdAsync(int Id)
             => await _context.DoctorWorkTimes.FindAsync(Id);

        public async Task<int> RemoveAsync(DoctorWorkTime doctorWorkTime)
        {
             _context.DoctorWorkTimes.Remove(doctorWorkTime);
            return await _context.SaveChangesAsync();
        }
    }
}
