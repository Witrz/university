using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace AspNetCoreMvc.Data.Migrations
{
    public partial class week10 : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "Like",
                table: "DiscussionForum",
                type: "int",
                nullable: false,
                defaultValue: 0);
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Like",
                table: "DiscussionForum");
        }
    }
}
