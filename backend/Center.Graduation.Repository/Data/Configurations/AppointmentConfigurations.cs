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
    public class AppointmentConfigurations : IEntityTypeConfiguration<Appointment>
    {
        public void Configure(EntityTypeBuilder<Appointment> builder)
        {
            //builder.HasKey(builder.)
            builder.HasOne(e => e.Doctor)
                .WithMany()
                .HasForeignKey(e => e.DoctorId)
                 .OnDelete(DeleteBehavior.Restrict); // أهم سطر

            builder.HasOne(e => e.Patient)
                .WithMany()
                .HasForeignKey(e => e.PatientId)
                 .OnDelete(DeleteBehavior.Restrict); // أهم سطر

            builder.Property(e => e.Time).IsRequired();
        }
    }
}
