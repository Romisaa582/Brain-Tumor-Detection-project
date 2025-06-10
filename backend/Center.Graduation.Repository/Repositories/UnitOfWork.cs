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
    public class UnitOfWork : IUnitOfWork
    {
        private readonly ApplicationDbContext _Context;
        private IDepartmentRepository _departmentRepository;
        private IDoctorWorkTimeRepository _doctorWorkTimeRepository;
        private IAppointmentRepository _appointmentRepository;


        public UnitOfWork(ApplicationDbContext context)
        {
            _Context = context;
            _departmentRepository = new DepartmentRepository(_Context);
            _doctorWorkTimeRepository = new DoctorWorkTimeRepository(_Context);
            _appointmentRepository = new AppointmentRepository(_Context);
        }



        public IDepartmentRepository departmentRepository => _departmentRepository;
        public IDoctorWorkTimeRepository doctorWorkTimeRepository => _doctorWorkTimeRepository;
        public IAppointmentRepository appointmentRepository => _appointmentRepository;
    }
}
