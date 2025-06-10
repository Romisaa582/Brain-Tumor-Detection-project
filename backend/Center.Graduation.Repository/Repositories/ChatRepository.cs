using Center.Graduation.Core.Entities;
using Center.Graduation.Core.Repositories;
using Center.Graduation.Repository.Contexts;
using Center.Graduation.Repository.RealTime;
using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Repository.Repositories
{
    public class ChatRepository:IChatRepository
    {
        private readonly ApplicationDbContext _context;

        public ChatRepository(ApplicationDbContext context)
        {
            _context = context;
        }
        public async Task<int> SendMessageAsync(ChatMessage chatMessage)
        {
            await _context.ChatMessages.AddAsync(chatMessage);
            return await _context.SaveChangesAsync();
        }

        public async Task<List<ChatMessage>> GetChatHistoryAsync(string receiverId, string senderId)
            => await _context.ChatMessages.Include(u => u.Sender).Include(u => u.Receiver).Where(e => (e.SenderId == senderId && e.ReceiverId == receiverId) ||
                                                      (e.SenderId == receiverId && e.ReceiverId == senderId))
                                                      .OrderBy(t => t.Timestamp)
                                                      .ToListAsync();

        public async Task<int> DeleteAsync(ChatMessage chatMessage)
        {
            _context.ChatMessages.Remove(chatMessage);
            return await _context.SaveChangesAsync();
        }

        public async Task<ChatMessage> GetMessageAsync(int MessageId)
            => await _context.ChatMessages.FindAsync(MessageId);

        public async Task<IEnumerable<ApplicationUser>> GetContactedUserAsync(string userId)
        {
           var userIds = await _context.ChatMessages
           .Where(m => m.SenderId == userId || m.ReceiverId == userId)
           .Select(m => m.SenderId == userId ? m.ReceiverId : m.SenderId)
           .Distinct()
           .ToListAsync();

            var contactedUsers = await _context.Users
                .Where(u => userIds.Contains(u.Id))
                .AsNoTracking()
                .ToListAsync();

            return contactedUsers;
        }
    }
}
