using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using AspNetCoreMvc.Models;

namespace AspNetCoreMvc.Data
{
    public class ApplicationDbContext : IdentityDbContext
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
            : base(options)
        {
        }
        public DbSet<AspNetCoreMvc.Models.DiscussionForum>? DiscussionForum { get; set; }
        public DbSet<AspNetCoreMvc.Models.Tutorial>? Tutorial { get; set; }
    }
}