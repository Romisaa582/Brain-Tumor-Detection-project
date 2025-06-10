using Center.Graduation.Core.Entities;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Repository.Contexts
{
    public class ApplicationDbContext:IdentityDbContext<ApplicationUser>
    {

        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options) : base(options)
        {

        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {

            modelBuilder.ApplyConfigurationsFromAssembly(Assembly.GetExecutingAssembly());
            base.OnModelCreating(modelBuilder);
        }


        public DbSet<Department> Departments { get; set; }
        public DbSet<DoctorWorkTime> DoctorWorkTimes { get; set; }
        public DbSet<Appointment> Appointments { get; set; }
        public DbSet<ChatMessage> ChatMessages { get; set; }
    }
}