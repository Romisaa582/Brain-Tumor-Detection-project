using Center.Graduation.API.Mapping;
using Center.Graduation.Core.Entities;
using Center.Graduation.Core.Repositories;
using Center.Graduation.Core.Services;
using Center.Graduation.Repository.Contexts;
using Center.Graduation.Repository.Repositories;
using Center.Graduation.Services;
using Microsoft.AspNetCore.Identity;

namespace Center.Graduation.API.Extension
{
    public static class ApplicationServicesEvtention
    {
        public static IServiceCollection AddApplicationServices(this IServiceCollection Services, IConfiguration configuration)
        {
            Services.AddScoped<IChatRepository, ChatRepository>();
            Services.AddScoped<ITokenService, TokenServices>();
            Services.AddScoped<IUnitOfWork, UnitOfWork>();

            Services.AddAutoMapper(M => M.AddProfile(new ApplicationProfile(configuration)));

            Services.AddIdentity<ApplicationUser, IdentityRole>()
                            .AddEntityFrameworkStores<ApplicationDbContext>();
            Services.AddSignalR();


            return Services;

        }
    }
}