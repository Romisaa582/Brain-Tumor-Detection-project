import * as signalR from '@microsoft/signalr';

class SignalRService {
  constructor() {
    this.connection = null;
    this.messageListeners = new Set();
    this.connectionPromise = null;
    this.retryCount = 0;
    this.maxRetries = 5;
    this.retryDelay = 2000; // Start with 2 seconds
  }

  async start() {
    if (this.connection?.state === signalR.HubConnectionState.Connected) {
      return;
    }

    if (this.connectionPromise) {
      return this.connectionPromise;
    }

    this.connectionPromise = new Promise(async (resolve, reject) => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          throw new Error('No authentication token found');
        }

        this.connection = new signalR.HubConnectionBuilder()
          .withUrl('https://tumortraker12.runasp.net/chatHub', {
            accessTokenFactory: () => token
          })
          .withAutomaticReconnect({
            nextRetryDelayInMilliseconds: retryContext => {
              if (retryContext.previousRetryCount === 0) {
                return 0; // If first retry, try immediately
              }
              
              // Exponential backoff with max delay of 30 seconds
              const delay = Math.min(1000 * Math.pow(2, retryContext.previousRetryCount - 1), 30000);
              return delay;
            }
          })
          .configureLogging(signalR.LogLevel.Information)
          .build();

        // Set up message handler
        this.connection.on('ReceiveMessage', (senderId, message) => {
          this.messageListeners.forEach(listener => listener(senderId, message));
        });

        // Handle connection events
        this.connection.onreconnecting(error => {
          console.log('Reconnecting to SignalR hub...', error);
        });

        this.connection.onreconnected(connectionId => {
          console.log('Reconnected to SignalR hub.', connectionId);
        });

        this.connection.onclose(error => {
          console.log('SignalR connection closed.', error);
          if (this.retryCount < this.maxRetries) {
            this.retryCount++;
            const delay = Math.min(this.retryDelay * Math.pow(2, this.retryCount - 1), 30000);
            setTimeout(() => this.start(), delay);
          }
        });

        await this.connection.start();
        this.retryCount = 0;
        this.connectionPromise = null;
        resolve();
      } catch (error) {
        console.error('Error starting SignalR connection:', error);
        this.connectionPromise = null;
        if (this.retryCount < this.maxRetries) {
          this.retryCount++;
          const delay = Math.min(this.retryDelay * Math.pow(2, this.retryCount - 1), 30000);
          setTimeout(() => this.start(), delay);
        }
        reject(error);
      }
    });

    return this.connectionPromise;
  }

  async stop() {
    if (this.connection?.state === signalR.HubConnectionState.Connected) {
      await this.connection.stop();
      this.connection = null;
    }
  }

  addMessageListener(callback) {
    this.messageListeners.add(callback);
    return () => this.messageListeners.delete(callback);
  }

  async sendMessage(receiverId, message) {
    if (!this.connection || this.connection.state !== signalR.HubConnectionState.Connected) {
      await this.start();
    }
    await this.connection.invoke('SendMessage', receiverId, message);
  }
}

export const signalRService = new SignalRService(); 