using Center.Graduation.Core.Entities;
using System.Net.Mail;
using System.Net;

namespace Center.Graduation.API.Helper
{
    public static class EmailSettings
    {
        public static void SendEmail(Emails email)
        {
            var client = new SmtpClient("smtp.gmail.com", 587);
            client.EnableSsl = true;
            client.Credentials = new NetworkCredential("zeyadenab220@gmail.com", "oipijnpwsjcwatrr");
            client.Send("zeyadenab220@gmail.com", email.To, email.Subject, email.Body);
        }
    }
}