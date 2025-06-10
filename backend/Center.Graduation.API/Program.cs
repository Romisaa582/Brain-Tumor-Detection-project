
using Center.Graduation.API.Extension;
using Center.Graduation.API.Helper;
using Center.Graduation.Repository.Contexts;
using Center.Graduation.Repository.RealTime;
using Microsoft.EntityFrameworkCore;
using System.Text.Json.Serialization;

namespace Center.Graduation.API
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var MyAllowSpecificOrigins = "_myAllowSpecificOrigins";

            var builder = WebApplication.CreateBuilder(args);
            // Configure CORS
            builder.Services.AddCors(options =>
            {
                options.AddPolicy(name: MyAllowSpecificOrigins,
                                  policy =>
                                  {
                                      policy.WithOrigins(
                                          "http://localhost:3000",
                                          "http://localhost:5173",
                                          "http://localhost:5173/",
                                          "http://localhost:5175",
                                          "https://tumortrackerfrontendproject-up4q.vercel.app/",
                                          "https://tumortrackerfrontendproject-up4q.vercel.app",
                                          "https://tumortraker12.runasp.net/",
                                          "http://tumortraker12.runasp.net"
                                      );
                                      policy.AllowAnyMethod();
                                      policy.AllowAnyHeader();
                                      policy.AllowCredentials();
                                  });
            });

            // Add SignalR
            builder.Services.AddSignalR();

            // Add Controllers
            builder.Services.AddControllers()
                  .AddJsonOptions(options =>
                  {
                      options.JsonSerializerOptions.Converters.Add(new JsonStringEnumConverter());
                      options.JsonSerializerOptions.Converters.Add(new TimeOnlyJsonConverter());
                  });

            // Swagger/OpenAPI
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            // Database Context
            builder.Services.AddDbContext<ApplicationDbContext>(options =>
            {
                options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection"));
            });

            // Custom Services and Auth
            builder.Services.AddApplicationServices(builder.Configuration);
            builder.Services.AddSwaggerGenJwtAuth();
            builder.Services.AddCustomJwtAuth(builder.Configuration);

            var app = builder.Build();

            // Middleware pipeline
            app.UseSwagger();
            app.UseSwaggerUI();

            app.UseStaticFiles();
            app.UseHttpsRedirection();

            app.UseCors(MyAllowSpecificOrigins);

            app.UseAuthorization();

            app.MapControllers();
            app.MapHub<ChatHub>("/chatHub");

            app.Run();
        }
    }
}
