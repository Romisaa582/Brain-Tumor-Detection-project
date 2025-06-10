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
    public class DepartmentRepository:IDepartmentRepository
    {

        private readonly ApplicationDbContext _context;

        public DepartmentRepository(ApplicationDbContext context)
        {
            _context = context;
        }


        public async Task<int> AddAsync(Department entity)
        {
            await _context.Departments.AddAsync(entity);
            return await _context.SaveChangesAsync();
        }
        public async Task<int> UpdateAsync(Department entity)
        {
            _context.Departments.Update(entity);
            return await _context.SaveChangesAsync();    
        }

        public async Task<int> DeleteAsync(Department entity)
        {
            _context.Departments.Remove(entity);
            return await _context.SaveChangesAsync();
        }

        public async Task<IEnumerable<Department>> GetAllAsync()
            => await _context.Departments.ToListAsync();

        public async Task<Department> GetByIdAsync(int Id)
            => await _context.Departments.FindAsync(Id);

        public async Task<IEnumerable<Department>> Search(string Name)
            => await _context.Departments.Where(d => d.Name == Name).ToListAsync();
    }
}
