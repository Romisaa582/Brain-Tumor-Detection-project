using System.ComponentModel.DataAnnotations;

namespace Center.Graduation.API.DTOs.ChatMessages
{
    public class SendMessageDTO
    {
        [Required(ErrorMessage = "ReciverId is required")]
        public string ReceiverId { get; set; }

        [Required(ErrorMessage = "Message is required")]
        public string Message { get; set; }
    }
}
