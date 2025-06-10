namespace Center.Graduation.API.DTOs.AccountDTO
{
    public class GetUserDTO
    {
        public string Id { get; set; }
        public string FirstName { get; set; }
        public string LastName { get; set; }
        public string UserName { get; set; }
        public string Email { get; set; }

        public string? PhotoURL { get; set; }
        public int Age { get; set; }

        public string? WebsiteURL { get; set; }
        public string Type { get; set; }
        public int? DepartmentId { get; set; }
    }
}
