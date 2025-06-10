import { useState, useEffect, useRef } from 'react';
import { signalRService } from '../services/signalRService';

const Chat = ({ doctorId, patientId, patientName, onClose, onMessageSent }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef(null);
  const token = localStorage.getItem('token');

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchWithAuth = async (url, options = {}) => {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          ...options.headers
        }
      });

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Unauthorized. Please log in again.');
        }
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.message || `Error: ${response.status}`);
      }

      return options.method === 'DELETE' ? null : await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  };

  const loadMessages = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await fetchWithAuth(
        `https://tumortraker12.runasp.net/api/Chat/GetChat?receiverId=${patientId}`
      );
      
      // Transform messages to match the API schema
      const formattedMessages = data.map(msg => ({
        id: msg.id,
        senderId: msg.senderId,
        senderName: msg.senderName,
        receiverId: msg.receiverId,
        receiverName: msg.receiverName,
        message: msg.message,
        timestamp: msg.timestamp
      }));

      setMessages(formattedMessages);
      setTimeout(scrollToBottom, 100);
    } catch (error) {
      console.error('Error loading messages:', error);
      if (error.message.includes('Unauthorized')) {
        setError('Session expired. Please log in again.');
      } else {
        setError('Failed to load messages. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let mounted = true;
    let retryCount = 0;
    const maxRetries = 3;

    const initialize = async () => {
      while (retryCount < maxRetries && mounted) {
        try {
          await loadMessages();
          break;
        } catch (error) {
          retryCount++;
          if (retryCount < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, 2000 * retryCount));
          }
        }
      }
    };

    initialize();

    // Listen for new messages
    const removeListener = signalRService.addMessageListener((senderId, message) => {
      if (senderId === patientId && mounted) {
        setMessages(prev => [...prev, {
          id: Date.now(), // Temporary ID until refresh
          senderId,
          receiverId: doctorId,
          message,
          timestamp: new Date().toISOString()
        }]);
        scrollToBottom();
      }
    });

    return () => {
      mounted = false;
      removeListener();
    };
  }, [patientId, doctorId, token]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!newMessage.trim() || sending) return;

    try {
      setSending(true);
      await fetchWithAuth('https://tumortraker12.runasp.net/api/Chat/send', {
        method: 'POST',
        body: JSON.stringify({
          receiverId: patientId,
          message: newMessage.trim()
        })
      });

      // Add message to local state
      setMessages(prev => [...prev, {
        id: Date.now(), // Temporary ID until refresh
        senderId: doctorId,
        receiverId: patientId,
        message: newMessage.trim(),
        timestamp: new Date().toISOString()
      }]);

      setNewMessage('');
      scrollToBottom();
      if (onMessageSent) onMessageSent();
    } catch (error) {
      console.error('Error sending message:', error);
      if (error.message.includes('Unauthorized')) {
        alert('Session expired. Please log in again.');
      } else {
        alert('Failed to send message. Please try again.');
      }
    } finally {
      setSending(false);
    }
  };

  const deleteMessage = async (messageId) => {
    try {
      await fetchWithAuth(`https://tumortraker12.runasp.net/api/Chat?MessageId=${messageId}`, {
        method: 'DELETE'
      });

      // Remove message from local state
      setMessages(prev => prev.filter(msg => msg.id !== messageId));
    } catch (error) {
      console.error('Error deleting message:', error);
      alert('Failed to delete message. Please try again.');
    }
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl h-[80vh] flex flex-col">
        {/* Chat Header */}
        <div className="p-4 border-b flex items-center justify-between bg-indigo-600 rounded-t-lg">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center">
              <span className="text-indigo-600 font-semibold text-lg">
                {patientName[0]}
              </span>
            </div>
            <h2 className="text-lg font-semibold text-white">{patientName}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-indigo-700 rounded-full transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {loading ? (
            <div className="flex justify-center items-center h-full">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
            </div>
          ) : error ? (
            <div className="text-center text-red-500 py-4">
              {error}
              <button
                onClick={loadMessages}
                className="block mx-auto mt-2 text-indigo-600 hover:text-indigo-700"
              >
                Try Again
              </button>
            </div>
          ) : messages.length === 0 ? (
            <div className="text-center text-gray-500 py-4">
              No messages yet. Start the conversation!
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.senderId === doctorId ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[70%] rounded-lg p-3 relative group ${
                    message.senderId === doctorId
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  <p className="break-words">{message.message}</p>
                  <p className={`text-xs mt-1 ${
                    message.senderId === doctorId
                      ? 'text-indigo-200'
                      : 'text-gray-500'
                  }`}>
                    {formatTime(message.timestamp)}
                  </p>
                  {message.senderId === doctorId && (
                    <button
                      onClick={() => deleteMessage(message.id)}
                      className="absolute -right-6 top-0 p-1 text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Message Input */}
        <form onSubmit={sendMessage} className="p-4 border-t">
          <div className="flex gap-2">
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder="Type a message..."
              className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <button
              type="submit"
              disabled={!newMessage.trim() || sending}
              className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {sending ? 'Sending...' : 'Send'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Chat; 