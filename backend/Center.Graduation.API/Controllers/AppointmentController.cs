using AutoMapper;
using Center.Graduation.API.DTOs.Appointment;
using Center.Graduation.API.Errors;
using Center.Graduation.Core.Entities;
using Center.Graduation.Core.Repositories;
using Center.Graduation.Repository.Contexts;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;

namespace Center.Graduation.API.Controllers
{
    public class AppointmentController : BaseController
    {
        private readonly IMapper _mapper;
        private readonly IUnitOfWork _unitOfWork;
        private readonly ApplicationDbContext _context;
        private readonly UserManager<ApplicationUser> _userManager;

        public AppointmentController(IMapper mapper, IUnitOfWork unitOfWork, ApplicationDbContext context , UserManager<ApplicationUser> userManager)
        {
            _mapper = mapper;
            _unitOfWork = unitOfWork;
            _context = context;
            _userManager = userManager;
        }

        [Authorize]
        [HttpPost]
        public async Task<ActionResult> BookAppointmentAsync(AppointmentCreateDTO dto)
        {

            // Get Doctor Work Time for that day
            var workTime = await _unitOfWork.doctorWorkTimeRepository.CheckDoctorWorkTime(dto.DoctorId, dto.DayOfWeek, dto.Time);

            if (workTime is null)
                return BadRequest("Doctor does not work at this time.");


            // Check for conflict with existing appointments
            var isTaken = await _unitOfWork.appointmentRepository.CheckAppointMent(dto.DoctorId , dto.DayOfWeek , dto.Time);

            if (isTaken)
                return BadRequest("This appointment slot is already taken.");


            var PatientId = User.FindFirstValue(ClaimTypes.NameIdentifier); // Get logged-in user ID

            // Everything is okay, create the appointment
            var appointment = new Appointment
            {
                DoctorId = dto.DoctorId,
                PatientId = PatientId,
                DayOfWeek = dto.DayOfWeek,
                Time = dto.Time
            };
             var count = await _unitOfWork.appointmentRepository.AddAsync(appointment);
            if(count > 0)
            return Ok("Appointment booked successfully.");

            return BadRequest("Error in Save");
        }

        [Authorize]
        [HttpGet("GetDoctorAppointment")]
        public async Task<ActionResult<IEnumerable<GetAppointment>>> GetDoctorAppointments(string DoctorId) 
        {
            var Doctor = await _userManager.FindByIdAsync(DoctorId);
            if(Doctor is null) 
                return NotFound("Doctor With this Id is not found");

            var Appoinments = await _unitOfWork.appointmentRepository.GetAllAppointmentOfDoctor(DoctorId);
            var map = _mapper.Map<IEnumerable<GetAppointment>>(Appoinments);
            return Ok(map);
        }


        [Authorize]
        [HttpGet("GetPatientAppointment")]
        public async Task<ActionResult<IEnumerable<GetAppointment>>> GetPatientAppointments(string PatientId)
        {
            var Doctor = await _userManager.FindByIdAsync(PatientId);
            if (Doctor is null)
                return NotFound("Patient With this Id is not found");

            var Appoinments = await _unitOfWork.appointmentRepository.GetAllAppointmentOfDoctor(PatientId);
            var map = _mapper.Map<IEnumerable<GetAppointment>>(Appoinments);
            return Ok(map);
        }


        [Authorize]
        [HttpGet("GetPatientAppointmentsWithDoctor")]
        public async Task<ActionResult<IEnumerable<GetAppointment>>> GetPatientAppointmentsWithDoctor(string DoctorId ,string PatientId)
        {
            var Doctor = await _userManager.FindByIdAsync(DoctorId);
            if (Doctor is null)
                return NotFound("Patient With this Id is not found");

            var Patient = await _userManager.FindByIdAsync(PatientId);
            if (Doctor is null)
                return NotFound("Doctor With this Id is not found");

            var Appoinments = await _unitOfWork.appointmentRepository.GatAllPatientAppointmentWithDoctor( DoctorId, PatientId);
            var map = _mapper.Map<IEnumerable<GetAppointment>>(Appoinments);
            return Ok(map);
        }

        [Authorize]
        [HttpGet("{id:int}")]
        public async Task<ActionResult<Appointment>> GetById(int Id) 
        {
            var appointment = await _unitOfWork.appointmentRepository.GetByIdAsync(Id);
            if (appointment is null)
                return NotFound("Appointment with this id is not found");
            return Ok(appointment);
        }


        [Authorize]
        [HttpDelete("{id:int}")]
        public async Task<ActionResult> Delete(int id)
        {
            var Appointment = await _unitOfWork.appointmentRepository.GetByIdAsync(id);
            if (Appointment is null)
                return NotFound(new ApiErrorResponse(StatusCodes.Status404NotFound));


            var DoctorId = User.FindFirstValue(ClaimTypes.NameIdentifier); // Get logged-in user ID

            if (Appointment.DoctorId == DoctorId)
            {
                var count = await _unitOfWork.appointmentRepository.RemoveAsync(Appointment);
                if (count > 0)
                    return Ok("Appointment deleted successfully.");

                return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest));
            }
            return BadRequest("Don't have access to remove this appointment");

         
        }
    }
}
