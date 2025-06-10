using AutoMapper;
using Center.Graduation.API.DTOs.AccountDTO;
using Center.Graduation.API.DTOs.ChatMessages;
using Center.Graduation.API.Errors;
using Center.Graduation.Core.Entities;
using Center.Graduation.Core.Repositories;
using Center.Graduation.Repository.RealTime;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.SignalR;
using System.Security.Claims;

namespace Center.Graduation.API.Controllers
{
    public class ChatController : BaseController
    {
        private readonly IHubContext<ChatHub> _hubContext;
        private readonly IChatRepository _ChatRepository;
        private readonly IMapper _mapper;

        public ChatController(IHubContext<ChatHub> hubContext, IChatRepository chatRepository, IMapper mapper)
        {
            _hubContext = hubContext;
            _ChatRepository = chatRepository;
            _mapper = mapper;
        }

        // Send a message via HTTP API
        [Authorize]
        [HttpPost("send")]
        public async Task<ActionResult> SendMessage([FromBody] SendMessageDTO request)
        {
            var senderId = User.FindFirstValue(ClaimTypes.NameIdentifier); // Get logged-in user ID

            if (senderId is null)
                return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest, "Invalid to get sender Id please sure a sign in "));

            var chatMessage = new ChatMessage   //Mapping
            {
                SenderId = senderId,
                ReceiverId = request.ReceiverId,
                Message = request.Message,
                Timestamp = DateTime.UtcNow
            };

            var count = await _ChatRepository.SendMessageAsync(chatMessage);

            if (count > 0)
            {
                // Send message via SignalR if receiver is online
                await _hubContext.Clients.User(request.ReceiverId)
                    .SendAsync("ReceiveMessage", senderId, request.Message);
                return Ok();
            }
            return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest, "Error in save message please try again"));

        }

        [Authorize]
        [HttpGet("GetContactUsers")]
        public async Task<ActionResult<IEnumerable<GetContactUser>>> GetContactUsers()
        {

            var UserId = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (UserId is null)
                return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest, "Invalid to get Users please sure a sign in "));

            var users = await _ChatRepository.GetContactedUserAsync(UserId);

            var baseUrl = $"{Request.Scheme}://{Request.Host}";
            var map = _mapper.Map<IEnumerable<GetContactUser>>(users, opt =>
            {
                opt.Items["BaseUrl"] = baseUrl;
            });
            return Ok(map);
        }

        [Authorize]
        [HttpGet("GetChat")]
        public async Task<ActionResult<IEnumerable<ReturnChat>>> GetChatHistory(string receiverId)
        {
            var senderId = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (senderId is null)
                return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest, "Invalid to get sender Id please sure a sign in "));

            var messages = await _ChatRepository.GetChatHistoryAsync(receiverId, senderId);
            var map = _mapper.Map<IEnumerable<ReturnChat>>(messages);

            return Ok(map);

        }


        [Authorize]
        [HttpDelete]
        public async Task<ActionResult> DeleteMessage(int MessageId)
        {
            var message = await _ChatRepository.GetMessageAsync(MessageId);
            if (message is null)
                return NotFound(new ApiErrorResponse(StatusCodes.Status404NotFound, "Message with this Id is not found"));

            var senderId = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (senderId != message.SenderId)
                return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest, "Don't have an access to remove this message"));

            var count = await _ChatRepository.DeleteAsync(message);
            if (count > 0)
            {
                return Ok();
            }
            return BadRequest(new ApiErrorResponse(StatusCodes.Status400BadRequest, "Error in delete please try again"));
        }
    }
}
