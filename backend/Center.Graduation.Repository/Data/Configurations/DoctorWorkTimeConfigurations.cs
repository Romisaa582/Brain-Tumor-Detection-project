using Center.Graduation.Core.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Repository.Data.Configurations
{
    public class DoctorWorkTimeConfigurations : IEntityTypeConfiguration<DoctorWorkTime>
    {
        public void Configure(EntityTypeBuilder<DoctorWorkTime> builder)
        {
            builder.HasOne(e => e.Doctor)
              .WithMany()
              .HasForeignKey(e => e.DoctorId)
              .OnDelete(DeleteBehavior.Restrict);



            builder.Property(p => p.Day).IsRequired();
            builder.Property(p => p.StartTime).IsRequired();
            builder.Property(p => p.EndTime).IsRequired();
            builder.Property(p => p.DoctorId).IsRequired();
        }
    }
}
