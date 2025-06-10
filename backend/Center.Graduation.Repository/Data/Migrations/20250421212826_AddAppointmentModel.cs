using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Center.Graduation.Repository.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddAppointmentModel : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_ChatMessages_AspNetUsers_ReceiverId",
                table: "ChatMessages");

            migrationBuilder.DropForeignKey(
                name: "FK_ChatMessages_AspNetUsers_SenderId",
                table: "ChatMessages");

            migrationBuilder.DropForeignKey(
                name: "FK_DoctorWorkTimes_AspNetUsers_DoctorId",
                table: "DoctorWorkTimes");

            migrationBuilder.AddColumn<string>(
                name: "ApplicationUserId",
                table: "DoctorWorkTimes",
                type: "nvarchar(450)",
                nullable: true);

            migrationBuilder.CreateTable(
                name: "Appointments",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    DoctorId = table.Column<string>(type: "nvarchar(450)", nullable: false),
                    DayOfWeek = table.Column<int>(type: "int", nullable: false),
                    Time = table.Column<TimeOnly>(type: "time", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Appointments", x => x.Id);
                    table.ForeignKey(
                        name: "FK_Appointments_AspNetUsers_DoctorId",
                        column: x => x.DoctorId,
                        principalTable: "AspNetUsers",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateIndex(
                name: "IX_DoctorWorkTimes_ApplicationUserId",
                table: "DoctorWorkTimes",
                column: "ApplicationUserId");

            migrationBuilder.CreateIndex(
                name: "IX_Appointments_DoctorId",
                table: "Appointments",
                column: "DoctorId");

            migrationBuilder.AddForeignKey(
                name: "FK_ChatMessages_AspNetUsers_ReceiverId",
                table: "ChatMessages",
                column: "ReceiverId",
                principalTable: "AspNetUsers",
                principalColumn: "Id",
                onDelete: ReferentialAction.Restrict);

            migrationBuilder.AddForeignKey(
                name: "FK_ChatMessages_AspNetUsers_SenderId",
                table: "ChatMessages",
                column: "SenderId",
                principalTable: "AspNetUsers",
                principalColumn: "Id",
                onDelete: ReferentialAction.Restrict);

            migrationBuilder.AddForeignKey(
                name: "FK_DoctorWorkTimes_AspNetUsers_ApplicationUserId",
                table: "DoctorWorkTimes",
                column: "ApplicationUserId",
                principalTable: "AspNetUsers",
                principalColumn: "Id");

            migrationBuilder.AddForeignKey(
                name: "FK_DoctorWorkTimes_AspNetUsers_DoctorId",
                table: "DoctorWorkTimes",
                column: "DoctorId",
                principalTable: "AspNetUsers",
                principalColumn: "Id",
                onDelete: ReferentialAction.Restrict);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_ChatMessages_AspNetUsers_ReceiverId",
                table: "ChatMessages");

            migrationBuilder.DropForeignKey(
                name: "FK_ChatMessages_AspNetUsers_SenderId",
                table: "ChatMessages");

            migrationBuilder.DropForeignKey(
                name: "FK_DoctorWorkTimes_AspNetUsers_ApplicationUserId",
                table: "DoctorWorkTimes");

            migrationBuilder.DropForeignKey(
                name: "FK_DoctorWorkTimes_AspNetUsers_DoctorId",
                table: "DoctorWorkTimes");

            migrationBuilder.DropTable(
                name: "Appointments");

            migrationBuilder.DropIndex(
                name: "IX_DoctorWorkTimes_ApplicationUserId",
                table: "DoctorWorkTimes");

            migrationBuilder.DropColumn(
                name: "ApplicationUserId",
                table: "DoctorWorkTimes");

            migrationBuilder.AddForeignKey(
                name: "FK_ChatMessages_AspNetUsers_ReceiverId",
                table: "ChatMessages",
                column: "ReceiverId",
                principalTable: "AspNetUsers",
                principalColumn: "Id");

            migrationBuilder.AddForeignKey(
                name: "FK_ChatMessages_AspNetUsers_SenderId",
                table: "ChatMessages",
                column: "SenderId",
                principalTable: "AspNetUsers",
                principalColumn: "Id");

            migrationBuilder.AddForeignKey(
                name: "FK_DoctorWorkTimes_AspNetUsers_DoctorId",
                table: "DoctorWorkTimes",
                column: "DoctorId",
                principalTable: "AspNetUsers",
                principalColumn: "Id");
        }
    }
}
