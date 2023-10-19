using FinalAssignment.Data;
using FinalAssignment.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Diagnostics;

namespace FinalAssignment.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;

        private readonly ApplicationDbContext _context;

        public HomeController(ILogger<HomeController> logger, ApplicationDbContext context)
        {
            _logger = logger;
            _context = context;
        }

        public async Task<IActionResult> Index()
        {
            return _context.GenAI != null ?
                          View(await _context.GenAI.OrderByDescending(i => i.Like).ToListAsync()) :
                          Problem("Entity set 'ApplicationDbContext.GenAI'  is null.");
        }

        public IActionResult Contact()
        {
            return View();
        }

        public IActionResult Jobs()
        {
            return View();
        }

        public IActionResult GenAI()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}