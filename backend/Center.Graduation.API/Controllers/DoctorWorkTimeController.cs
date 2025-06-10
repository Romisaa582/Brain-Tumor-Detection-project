using AutoMapper;
using Center.Graduation.API.DTOs.Department;
using Center.Graduation.API.DTOs.DoctorWorkTime;
using Center.Graduation.API.Errors;
using Center.Graduation.API.Helper;
using Center.Graduation.Core.Entities;
using Center.Graduation.Core.Repositories;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace Center.Graduation.API.Controllers
{
    public class DoctorWorkTimeController : BaseController
    {
        private readonly IMapper _mapper;
        private readonly IUnitOfWork _unitOfWork;
        private readonly UserManager<ApplicationUser> _userManager;

        public DoctorWorkTimeController(IMapper mapper, IUnitOfWork unitOfWork , UserManager<ApplicationUser> userManager)
        {
            _mapper = mapper;
            _unitOfWork = unitOfWork;
            _userManager = userManager;
        }


        [Authorize]
        [HttpPost]
        public async Task<ActionResult> Add(AddDoctorWorkTimeDTO DoctorWorkTime) 
        {
            if (!ModelState.IsValid)
                return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest));



            var WorkTime = _mapper.Map<DoctorWorkTime>(DoctorWorkTime);

            WorkTime.DoctorId = User.FindFirstValue(ClaimTypes.NameIdentifier);
            var count = await _unitOfWork.doctorWorkTimeRepository.AddAsync(WorkTime);

            if (count > 0)
                return Ok(WorkTime);

            return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest));

        }


        [Authorize]
        [HttpDelete]
        public async Task<ActionResult> Delete(int id) 
        {
            var worktime = await _unitOfWork.doctorWorkTimeRepository.GetByIdAsync(id);
            if (worktime is null)
                return NotFound(new ApiErrorResponse(StatusCodes.Status404NotFound, "Doctor Work Time with this Id is not found"));


            var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);

            if (worktime.DoctorId == userId || User.IsInRole("Admin"))
            {
                var count = await _unitOfWork.doctorWorkTimeRepository.RemoveAsync(worktime);
                if (count > 0)
                {
                    return Ok();
                }
                return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest, "Error in Delete WorkTime"));
            }
            return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest, "Don't have an access to remove this WorkTime"));
        }

        [HttpGet]
        public async Task<ActionResult<IEnumerable<ReturnDoctorWorkTime>>> GetAllDoctorWorkTime(string DoctorId) 
        {
            var doctor = await _userManager.FindByIdAsync(DoctorId);
            if (doctor is null)
                return NotFound(new ApiErrorResponse(StatusCodes.Status404NotFound, "User with this Id is not found"));

            var doctorWorkTime = await _unitOfWork.doctorWorkTimeRepository.GetAllDoctorWorkTime(DoctorId);

            var result = _mapper.Map<IReadOnlyList<ReturnDoctorWorkTime>>(doctorWorkTime);
            return Ok(result);
        }

    }
}
