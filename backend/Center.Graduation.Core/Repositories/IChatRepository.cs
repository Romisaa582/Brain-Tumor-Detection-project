using Center.Graduation.Core.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Core.Repositories
{
    public interface IChatRepository
    {
        Task<List<ChatMessage>> GetChatHistoryAsync(string receiverId, string senderId);
        Task<ChatMessage> GetMessageAsync(int MessageId);
        Task<int> SendMessageAsync(ChatMessage chatMessage);
        Task<int> DeleteAsync(ChatMessage chatMessage);
        Task<IEnumerable<ApplicationUser>> GetContactedUserAsync(string UserId);
    }
}
