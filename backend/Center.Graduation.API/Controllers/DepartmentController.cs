using AutoMapper;
using Center.Graduation.API.DTOs;
using Center.Graduation.API.DTOs.Department;
using Center.Graduation.API.Errors;
using Center.Graduation.API.Helper;
using Center.Graduation.Core.Entities;
using Center.Graduation.Core.Repositories;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace Center.Graduation.API.Controllers
{
    public class DepartmentController : BaseController
    {
        private readonly IMapper _mapper;
        private readonly IUnitOfWork _unitOfWork;

        public DepartmentController(IMapper mapper, IUnitOfWork unitOfWork)
        {
            _mapper = mapper;
            _unitOfWork = unitOfWork;
        }
           //end points 
        [HttpGet]
        public async Task<ActionResult<IEnumerable<DepartmentDTO>>> GetAll()
        {
            var departments = await _unitOfWork.departmentRepository.GetAllAsync();
            var result = _mapper.Map<IReadOnlyList<DepartmentDTO>>(departments);
            return Ok(result);
        }

        [HttpGet("{id:int}")]
        public async Task<ActionResult<DepartmentDTO>> GetById(int id)
        {
            var department = await _unitOfWork.departmentRepository.GetByIdAsync(id);
            if (department == null)
                return NotFound(new ApiErrorResponse(StatusCodes.Status404NotFound));

            var result = _mapper.Map<DepartmentDTO>(department);
            return Ok(result);
        }

        [HttpPost]
        public async Task<ActionResult> Add([FromBody] DepartmentDTO departmentDto)
        {
            if (!ModelState.IsValid)
                return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest));

            var department = _mapper.Map<Department>(departmentDto);
            var count = await _unitOfWork.departmentRepository.AddAsync(department);

            if (count > 0)
                return Ok(departmentDto);

            return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest));
        }

        [HttpPut]
        public async Task<ActionResult> Update([FromBody] DepartmentDTO departmentDto)
        {
            if (!ModelState.IsValid)
                return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest));

            var department = _mapper.Map<Department>(departmentDto);
            var count = await _unitOfWork.departmentRepository.UpdateAsync(department);

            if (count > 0)
                return Ok(departmentDto);

            return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest));
        }

        [HttpDelete("{id:int}")]
        public async Task<ActionResult> Delete(int id)
        {
            var department = await _unitOfWork.departmentRepository.GetByIdAsync(id);
            if (department is null)
                return NotFound(new ApiErrorResponse(StatusCodes.Status404NotFound));

            var count = await _unitOfWork.departmentRepository.DeleteAsync(department);
            if (count > 0)
                return Ok("Department deleted successfully.");

            return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest));
        }
    }
}