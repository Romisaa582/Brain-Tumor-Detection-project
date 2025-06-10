using AutoMapper;
using Center.Graduation.API.DTOs.AccountDTO;
using Center.Graduation.API.Errors;
using Center.Graduation.API.Helper;
using Center.Graduation.Core.Entities;
using Center.Graduation.Core.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations;

namespace Center.Graduation.API.Controllers
{
    public class AccountController : BaseController
    {
        private readonly UserManager<ApplicationUser> _userManager; //
        private readonly SignInManager<ApplicationUser> _signInManager;
        private readonly ITokenService _tokenServices;
        private readonly IMapper _mapper;

        public AccountController(UserManager<ApplicationUser> userManager,
            SignInManager<ApplicationUser> signInManager,
            ITokenService tokenServices,
            IMapper mapper)
        {
            _userManager = userManager;
            _signInManager = signInManager;
            _tokenServices = tokenServices;
            _mapper = mapper;
        }




        [HttpGet("GetAllUsers")]
        public async Task<ActionResult<IEnumerable<GetUserDTO>>> GetAllUsers()
        {
            var users = await _userManager.Users.ToListAsync();
            var baseUrl = $"{Request.Scheme}://{Request.Host}";
            var map = _mapper.Map<IEnumerable<GetUserDTO>>(users, opt =>
            {
                opt.Items["BaseUrl"] = baseUrl;
            });
            return Ok(map);
        }

        [HttpGet("GetAllDoctors")]
        public async Task<ActionResult<IEnumerable<GetUserDTO>>> GetAllDoctors()
        {
            var users = await _userManager.GetUsersInRoleAsync("Doctor");
            var baseUrl = $"{Request.Scheme}://{Request.Host}";
            var map = _mapper.Map<IEnumerable<GetUserDTO>>(users, opt =>
            {
                opt.Items["BaseUrl"] = baseUrl;
            });
            return Ok(map);
        }

        [HttpGet("GetAllPatients")]
        public async Task<ActionResult<IEnumerable<GetUserDTO>>> GetAllPatients()
        {
            var users = await _userManager.GetUsersInRoleAsync("Patient");
            var baseUrl = $"{Request.Scheme}://{Request.Host}";
            var map = _mapper.Map<IEnumerable<GetUserDTO>>(users, opt =>
            {
                opt.Items["BaseUrl"] = baseUrl;
            });
            return Ok(map);
        }

        [HttpGet("GetAllAdmins")]
        public async Task<ActionResult<IEnumerable<GetUserDTO>>> GetAllAdmins()
        {
            var users = await _userManager.GetUsersInRoleAsync("Admin");
            var baseUrl = $"{Request.Scheme}://{Request.Host}";
            var map = _mapper.Map<IEnumerable<GetUserDTO>>(users, opt =>
            {
                opt.Items["BaseUrl"] = baseUrl;
            });
            return Ok(map);
        }

        [HttpGet("SearchByName")]
        public async Task<ActionResult<IEnumerable<GetUserDTO>>> SearchByName(string Name)
        {
            var users = await _userManager.Users.Where(u => u.NormalizedUserName.Contains(Name.ToUpper())).ToListAsync();
            var baseUrl = $"{Request.Scheme}://{Request.Host}";
            var map = _mapper.Map<IEnumerable<GetUserDTO>>(users, opt =>
            {
                opt.Items["BaseUrl"] = baseUrl;
            });
            return Ok(map);
        }

        [AllowAnonymous]
        [HttpGet("GetUserById")]
        public async Task<ActionResult<GetUserDTO>> GetUserById(string userId)
        {
            if (ModelState.IsValid)
            {
                var user = await _userManager.FindByIdAsync(userId);
                if (user is null)
                {
                    return NotFound(new ApiErrorResponse(StatusCodes.Status404NotFound, "User with this Id is not found"));
                }

                var baseUrl = $"{Request.Scheme}://{Request.Host}";
                var map = _mapper.Map<GetUserDTO>(user, opt =>
                {
                    opt.Items["BaseUrl"] = baseUrl;
                });
                return Ok(map);

            }
            return BadRequest(new ApiValidationResponse(400
                     , "a bad Request , You have made"
                     , ModelState.Values
                     .SelectMany(v => v.Errors)
                     .Select(e => e.ErrorMessage)
                     .ToList()));
        }

        [AllowAnonymous]
        [HttpPost("login")]
        public async Task<ActionResult<UserDTO>> Login(LoginDTO model)
        {
            if (ModelState.IsValid)
            {
                var user = await _userManager.FindByEmailAsync(model.Email);
                if (user == null) return Unauthorized(new ApiErrorResponse(401, "User with this Email is not found"));
                var result = await _signInManager.CheckPasswordSignInAsync(user, model.Password, false);
                if (!result.Succeeded) return Unauthorized(new ApiErrorResponse(401, "Password is InCorrect"));

                var userRole = await _userManager.GetRolesAsync(user);
                return Ok(new UserDTO()
                {
                    Id = user.Id,
                    UserName = user.UserName,
                    Role = userRole[0],
                    Email = user.Email,
                    PhotoURL = $"{Request.Scheme}://{Request.Host}/Images/{user.PhotoURL}",
                    Token = await _tokenServices.CreateTokenAsync(user, _userManager)
                });
            }
            return BadRequest(new ApiValidationResponse(400
               , "a bad Request , You have made"
               , ModelState.Values
               .SelectMany(v => v.Errors)
               .Select(e => e.ErrorMessage)
               .ToList()));
        }

        [AllowAnonymous]
        [HttpPost("DoctorRegister")]
        public async Task<ActionResult<UserDTO>> DoctorRegister(RegisterDoctorDTO model)
        {
            if (ModelState.IsValid)
            {
                if (CheckEmailExists(model.Email).Result.Value) // هنا بيشوف لو الايميل بتاعي موجود ولا لا
                {
                    return BadRequest(new ApiErrorResponse(400, "Email Is Already in Used"));
                }


                var user = _mapper.Map<ApplicationUser>(model);    //Map From AppUserDTO TO App User 

                if (model.Photo is not null)
                {
                    user.PhotoURL = DocumentSettings.Upload(model.Photo, "Images");
                }

                var Result = await _userManager.CreateAsync(user, model.Password);

                if (!Result.Succeeded)
                    return BadRequest(new ApiValidationResponse(StatusCodes.Status400BadRequest
                    , "a bad Request , You have made"
                    , Result.Errors.Select(e => e.Description).ToList())); //UnSaved

                await _userManager.AddToRoleAsync(user, "Doctor");

                var ReturnedUser = new UserDTO()
                {
                    Id = user.Id,
                    UserName = user.UserName,
                    Role = "Doctor",
                    Email = user.Email,
                    PhotoURL = $"{Request.Scheme}://{Request.Host}/Images/{user.PhotoURL}",
                    Token = await _tokenServices.CreateTokenAsync(user, _userManager)
                };
                return Ok(ReturnedUser);
            }
            return BadRequest(new ApiValidationResponse(400
                     , "a bad Request , You have made"
                     , ModelState.Values
                     .SelectMany(v => v.Errors)
                     .Select(e => e.ErrorMessage)
                     .ToList()));
        }

        [AllowAnonymous]
        [HttpPost("PatientRegister")]
        public async Task<ActionResult<UserDTO>> PatientRegister(RegisterPatientDTO model)
        {
            if (ModelState.IsValid)
            {
                if (CheckEmailExists(model.Email).Result.Value) // هنا بيشوف لو الايميل بتاعي موجود ولا لا
                {
                    return BadRequest(new ApiErrorResponse(400, "Email Is Already in Used"));
                }

                var user = _mapper.Map<ApplicationUser>(model);    //Map From AppUserDTO TO App User 

                if (model.Photo is not null)
                {
                    user.PhotoURL = DocumentSettings.Upload(model.Photo, "Images");
                }

                var Result = await _userManager.CreateAsync(user, model.Password);

                if (!Result.Succeeded)
                    return BadRequest(new ApiValidationResponse(StatusCodes.Status400BadRequest
                    , "a bad Request , You have made"
                    , Result.Errors.Select(e => e.Description).ToList())); //UnSaved

                await _userManager.AddToRoleAsync(user, "Patient");

                var ReturnedUser = new UserDTO()
                {
                    Id = user.Id,
                    UserName = user.UserName,
                    Role = "Patient",
                    Email = user.Email,
                    PhotoURL = $"{Request.Scheme}://{Request.Host}/Images/{user.PhotoURL}",
                    Token = await _tokenServices.CreateTokenAsync(user, _userManager)
                };
                return Ok(ReturnedUser);
            }
            return BadRequest(new ApiValidationResponse(400
                     , "a bad Request , You have made"
                     , ModelState.Values
                     .SelectMany(v => v.Errors)
                     .Select(e => e.ErrorMessage)
                     .ToList()));
        }

        [HttpGet("emailExists")]
        public async Task<ActionResult<bool>> CheckEmailExists(string Email)
        {
            var user = await _userManager.FindByEmailAsync(Email);
            if (user is null) return false;
            else return true;
        }

        [Authorize(Roles = "Admin")]
        [HttpDelete("DeleteUser")]
        public async Task<ActionResult> DeleteUser(string Id)
        {
            if (ModelState.IsValid)
            {
                var user = await _userManager.FindByIdAsync(Id);
                if (user is not null)
                {
                    var result = await _userManager.DeleteAsync(user);
                    if (result.Succeeded)
                    {
                        return Ok();
                    }
                    return BadRequest(new ApiValidationResponse(StatusCodes.Status400BadRequest
                        , "a bad Request , You have made"
                        , result.Errors.Select(e => e.Description).ToList()));
                }
                return NotFound(new ApiErrorResponse(404, "User with this id is not found"));
            }
            return BadRequest(new ApiValidationResponse(400
                , "a bad Request , You have made"
                , ModelState.Values
                .SelectMany(v => v.Errors)
                .Select(e => e.ErrorMessage)
                .ToList()));
        }

        [AllowAnonymous]
        [HttpPost("SendEmail")]
        public async Task<ActionResult> SendEmail([DataType(DataType.EmailAddress)] string Email)
        {
            if (ModelState.IsValid)
            {
                var user = await _userManager.FindByEmailAsync(Email);
                if (user != null)
                {
                    Random random = new Random();
                    int Code = random.Next(1000, 9999);
                    var email = new Emails()
                    {
                        To = Email,
                        Subject = "Reset Password",
                        Body = $"Resetting Your Password in Center App\r\n\r\nOpen the app and click \"Forgot Password?\r\n\r\nEnter your email or username.Code\r\n\r\nCode = {Code}"
                    };

                    EmailSettings.SendEmail(email);
                    return Ok(Code);
                }
                return NotFound(new ApiErrorResponse(404, "User with this Email is not found"));
            }
            return BadRequest(new ApiValidationResponse(400
                , "a bad Request , You have made"
                , ModelState.Values
                .SelectMany(v => v.Errors)
                .Select(e => e.ErrorMessage)
                .ToList()));
        }

        [AllowAnonymous]
        [HttpPut("ChangePassword")]
        public async Task<ActionResult> ChangePassword([FromBody] UpdatePasswordDTO updatePasswordDTO)
        {
            if (ModelState.IsValid)
            {
                var user = await _userManager.FindByEmailAsync(updatePasswordDTO.Email);
                if (user is not null)
                {
                    var result = await _userManager.RemovePasswordAsync(user);
                    if (result.Succeeded)
                    {
                        result = await _userManager.AddPasswordAsync(user, updatePasswordDTO.Password);
                        if (result.Succeeded)
                        {
                            return Ok("changed");
                        }
                    }
                    return BadRequest(new ApiValidationResponse(StatusCodes.Status400BadRequest
                        , "a bad Request , You have made"
                        , ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage).ToList()));
                }
                return NotFound(new ApiErrorResponse(StatusCodes.Status404NotFound, "User with this Email is not found"));
            }
            return BadRequest(new ApiValidationResponse(400
                , "a bad Request , You have made"
                , ModelState.Values
                .SelectMany(v => v.Errors)
                .Select(e => e.ErrorMessage)
                .ToList()));
        }

        [HttpGet("UserRoles")]
        public async Task<ActionResult> GetUserRoles(string Email)
        {
            if (ModelState.IsValid)
            {
                var user = await _userManager.FindByEmailAsync(Email);
                if (user != null)
                {
                    var roles = await _userManager.GetRolesAsync(user);
                    return Ok(roles);
                }
                return NotFound(new ApiValidationResponse(404, "User with this Email is not found"));
            }
            return BadRequest(new ApiValidationResponse(400,
                             "a bad Request , You have made",
                             ModelState.Values
                             .SelectMany(v => v.Errors)
                             .Select(e => e.ErrorMessage)
                             .ToList()));
        }

        [AllowAnonymous]
        [HttpPost("AddToRole")]
        public async Task<IActionResult> AddToRole([FromBody] AddToRoleDTO addToRoleDTO)
        {
            if (ModelState.IsValid)
            {
                var user = await _userManager.FindByEmailAsync(addToRoleDTO.Email);
                if (user is not null)
                {
                    var result = await _userManager.AddToRoleAsync(user, addToRoleDTO.Role);
                    if (result.Succeeded)
                        return Ok();

                    else
                        return BadRequest(new ApiValidationResponse(StatusCodes.Status400BadRequest
                                          , "a bad Request , You have made"
                                          , ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage).ToList()));

                }
                return NotFound(new ApiErrorResponse(404, "User with this Email is not found"));
            }
            return BadRequest(new ApiValidationResponse(StatusCodes.Status400BadRequest
                              , "a bad Request , You have made"
                              , ModelState.Values
                              .SelectMany(v => v.Errors)
                              .Select(e => e.ErrorMessage)
                              .ToList()));
        }
    }
}