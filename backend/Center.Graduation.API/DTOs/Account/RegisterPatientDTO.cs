using System.ComponentModel.DataAnnotations;

namespace Center.Graduation.API.DTOs.AccountDTO
{
    public class RegisterPatientDTO
    {

        [Required(ErrorMessage = "UserName is required")]
        public string UserName { get; set; }


        [Required(ErrorMessage = "FirstName is required")]

        public string FirstName { get; set; }

        [Required(ErrorMessage = "LastName is required")]
        public string LastName { get; set; }

        [Required]
        [EmailAddress]
        public string Email { get; set; }

        [Required(ErrorMessage = "Password is required")]
        public string Password { get; set; }

        public IFormFile Photo { get; set; }
        //public string? PhotoURL { get; set; }


        [Required(ErrorMessage = "Age of user is required")]
        public int Age { get; set; }

        public string? WebsiteURL { get; set; }

        [Required(ErrorMessage = "Type of user is required")]
        public string Type { get; set; }

    }
}
