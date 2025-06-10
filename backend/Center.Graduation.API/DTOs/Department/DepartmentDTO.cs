using System.ComponentModel.DataAnnotations;

namespace Center.Graduation.API.DTOs.Department
{
    public class DepartmentDTO
    {
        public int Id { get; set; }

        [Required(ErrorMessage = "DepartmentName is required")]
        public string Name { get; set; }
    }
}
