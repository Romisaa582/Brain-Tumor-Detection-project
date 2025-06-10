namespace Center.Graduation.API.DTOs.AccountDTO
{
    public class UserDTO
    {
        public string Id { get; set; }
        public string UserName { get; set; }
        public string Role { get; set; }
        public string Email { get; set; }
        public string? PhotoURL { get; set; }
        public string Token { get; set; }
    }
}
