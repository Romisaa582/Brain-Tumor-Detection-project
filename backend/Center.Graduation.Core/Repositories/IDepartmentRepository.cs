using Center.Graduation.Core.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Core.Repositories
{
    public interface IDepartmentRepository
    {
        Task<IEnumerable<Department>> GetAllAsync();
        Task<Department> GetByIdAsync(int Id);
        Task<int> AddAsync(Department entity);
        Task<int> UpdateAsync(Department entity);
        Task<int> DeleteAsync(Department entity);
        public Task<IEnumerable<Department>> Search(string Name);
    }
}