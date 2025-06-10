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
    public class ApplicationUserConfigurations : IEntityTypeConfiguration<ApplicationUser>
    {
        public void Configure(EntityTypeBuilder<ApplicationUser> builder)
        {

            builder.HasOne(e => e.Department)
                .WithMany()
                .HasForeignKey(e => e.DepartmentId).IsRequired(false);    //1 to m   fk m  set one 



            builder.HasIndex(u => u.Email).IsUnique(true);
            builder.Property(p => p.FirstName).IsRequired();
            builder.Property(p => p.LastName).IsRequired();
            builder.Property(p => p.Age).IsRequired();
            builder.Property(p => p.Type).IsRequired();
            builder.Property(p => p.WebsiteURL).IsRequired(false);
            builder.Property(p => p.DepartmentId).IsRequired(false);
        }
    }
}