using Microsoft.AspNetCore.SignalR;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Center.Graduation.Repository.RealTime
{
    public class ChatHub:Hub
    {
        private static readonly ConcurrentDictionary<string, string> _userConnections = new();

        // Method to send a message to a specific user
        public async Task SendMessageToUser(string senderUserId, string receiverUserId, string message)
        {
            if (_userConnections.TryGetValue(receiverUserId, out var connectionId))
            {
                await Clients.Client(connectionId).SendAsync("ReceiveMessage", senderUserId, receiverUserId, message);
            }
        }
        public override async Task OnDisconnectedAsync(Exception? exception)
        {
            var userId = _userConnections.FirstOrDefault(x => x.Value == Context.ConnectionId).Key;
            if (userId != null)
            {
                _userConnections.TryRemove(userId, out _);
                await Clients.All.SendAsync("UserDisconnected", userId);
            }
            await base.OnDisconnectedAsync(exception);
        }
    }
}
