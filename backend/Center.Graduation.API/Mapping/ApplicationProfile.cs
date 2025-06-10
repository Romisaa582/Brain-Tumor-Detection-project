using AutoMapper;
using Center.Graduation.API.DTOs;
using Center.Graduation.API.DTOs.AccountDTO;
using Center.Graduation.API.DTOs.Appointment;
using Center.Graduation.API.DTOs.ChatMessages;
using Center.Graduation.API.DTOs.Department;
using Center.Graduation.API.DTOs.DoctorWorkTime;
using Center.Graduation.Core.Entities;

namespace Center.Graduation.API.Mapping
{
    public class ApplicationProfile:Profile
    {
        public ApplicationProfile(IConfiguration configuration)
        {
            CreateMap<RegisterPatientDTO, ApplicationUser>().ReverseMap();
            CreateMap<ApplicationUser, GetUserDTO>()
                .ForMember(dest=>dest.PhotoURL , opt=>opt.MapFrom(
                    (src , dest , destMember , context)=> 
                    {
                        var baseUrl = context.Items["BaseUrl"] as string;
                        return string.IsNullOrEmpty(src.PhotoURL) ? null : $"{baseUrl}/Images/{src.PhotoURL}";

                    }));
            CreateMap<RegisterDoctorDTO, ApplicationUser>().ReverseMap();
            CreateMap<RegisterPatientDTO, ApplicationUser>().ReverseMap();
            CreateMap<Department, DepartmentDTO>().ReverseMap();
            CreateMap<DoctorWorkTime, ReturnDoctorWorkTime>()
                .ForMember(dest => dest.DoctorName, opt => opt.MapFrom(src => src.Doctor.UserName))
                .ForMember(dest => dest.DoctorId, opt => opt.MapFrom(src => src.Doctor.Id));


            CreateMap<AddDoctorWorkTimeDTO, DoctorWorkTime>().ReverseMap(); 


            CreateMap<ChatMessage, ReturnChat>()
                .ForMember(c => c.SenderName, opt => opt.MapFrom(src => src.Sender.UserName))
                .ForMember(c => c.ReceiverName, opt => opt.MapFrom(src => src.Receiver.UserName));


            CreateMap<Appointment, GetAppointment>()
                .ForMember(a => a.PatientName, opt => opt.MapFrom(src => src.Patient.UserName))
                .ForMember(a => a.DoctorName, opt => opt.MapFrom(src => src.Doctor.UserName));



            CreateMap<ApplicationUser, GetContactUser>()
                .ForMember(dest => dest.PhotoURL, opt => opt.MapFrom(
                    (src, dest, destMember, context) =>
                    {
                        var baseUrl = context.Items["BaseUrl"] as string;
                        return string.IsNullOrEmpty(src.PhotoURL) ? null : $"{baseUrl}/Images/{src.PhotoURL}";

                    }));
        }
    
    }
}
