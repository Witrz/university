using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;
using FinalAssignment.Data;
using FinalAssignment.Models;
using IHostingEnvironment = Microsoft.AspNetCore.Hosting.IHostingEnvironment;
using Microsoft.AspNetCore.Routing.Constraints;
using System.Data;
using System.Text.Json;

namespace FinalAssignment.Controllers
{
    public class GenAIsController : Controller
    {
        private readonly ApplicationDbContext _context;
        private readonly IHostingEnvironment _hostingEnv;

        public GenAIsController(ApplicationDbContext context, IHostingEnvironment hostingEnv)
        {
            _context = context;
            _hostingEnv = hostingEnv;
        }

        public async Task<IActionResult> IncreaseLike(int? id)
        {
            if (id == null)
            {
                return NotFound();
            }
            var genAI = await _context.GenAI.FindAsync(id);
            if (genAI == null)
            {
                return NotFound();
            }
            
            var value = HttpContext.Session.GetString(genAI.GenAIName);
            bool val = value == null ? false : JsonSerializer.Deserialize<bool>(value);


            if (ModelState.IsValid && (val == false))
            {
                try
                {
                    HttpContext.Session.SetString(genAI.GenAIName, JsonSerializer.Serialize(true));
                    genAI.Like++;
                    _context.Update(genAI);
                    await _context.SaveChangesAsync();
                }
                catch (DbUpdateConcurrencyException)
                {
                    if (!GenAIExists(genAI.Id))
                    {
                        return NotFound();
                    }
                    else
                    {
                        throw;
                    }
                }
                return RedirectToAction(nameof(Index));
            }
            return RedirectToAction(nameof(Index));
        }

        // GET: GenAIs
        public async Task<IActionResult> Index()
        {
              return _context.GenAI != null ? 
                          View(await _context.GenAI.OrderByDescending(i => i.Like).ToListAsync()) :
                          Problem("Entity set 'ApplicationDbContext.GenAI'  is null.");
        }

        // GET: GenAIs/Details/5
        public async Task<IActionResult> Details(int? id)
        {
            if (id == null || _context.GenAI == null)
            {
                return NotFound();
            }

            var genAI = await _context.GenAI
                .FirstOrDefaultAsync(m => m.Id == id);
            if (genAI == null)
            {
                return NotFound();
            }

            return View(genAI);
        }

        // GET: GenAIs/Create
        public IActionResult Create()
        {
            GenAI genAI = new GenAI();
            genAI.AnchorLink = "none";
            genAI.ImageFilename = "none";
            return View(genAI);
        }

        // POST: GenAIs/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Create([Bind("Id,GenAIName,Summary")] GenAI genAI, UploadFile uploadFile)
        {
            if(uploadFile.File != null)
            {
                var fileName = Path.GetFileName(uploadFile.File.FileName);
                var fileNameWOExt = Path.GetFileNameWithoutExtension(uploadFile.File.FileName);
                genAI.ImageFilename = fileNameWOExt;
                genAI.AnchorLink = "GenAIs/#" + fileNameWOExt.ToLower();
                var filePath = Path.Combine(_hostingEnv.WebRootPath, "images", fileName);
                using (var fileStream = new FileStream(filePath, FileMode.Create))
                {
                    await uploadFile.File.CopyToAsync(fileStream);
                }
            }
            
            if (ModelState.IsValid)
            {
                _context.Add(genAI);
                await _context.SaveChangesAsync();
                return RedirectToAction(nameof(Index));
            }
            return View(genAI);
        }

        // GET: GenAIs/Edit/5
        public async Task<IActionResult> Edit(int? id)
        {
            if (id == null || _context.GenAI == null)
            {
                return NotFound();
            }

            var genAI = await _context.GenAI.FindAsync(id);
            if (genAI == null)
            {
                return NotFound();
            }
            return View(genAI);
        }

        // POST: GenAIs/Edit/5
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Edit(int id, [Bind("Id,GenAIName,Summary, Like, AnchorLink, ImageFilename, uploadFile")] GenAI genAI, UploadFile uploadFile)
        {
            if (id != genAI.Id)
            {
                return NotFound();
            }
            
            if (uploadFile.File != null)
            {
                var fileName = Path.GetFileName(uploadFile.File.FileName);
                var fileNameWOExt = Path.GetFileNameWithoutExtension(uploadFile.File.FileName);
                genAI.ImageFilename = fileNameWOExt;
                genAI.AnchorLink = "GenAIs/#" + fileNameWOExt.ToLower();
                var filePath = Path.Combine(_hostingEnv.WebRootPath, "images", fileName);
                using (var fileStream = new FileStream(filePath, FileMode.Create))
                {
                    await uploadFile.File.CopyToAsync(fileStream);
                }
            }
            else
            {
                ModelState.ClearValidationState("File");
                ModelState.MarkFieldValid("File");
            }
            

            if (ModelState.IsValid)
            {
                try
                {
                    _context.Update(genAI);
                    await _context.SaveChangesAsync();
                }
                catch (DbUpdateConcurrencyException)
                {
                    if (!GenAIExists(genAI.Id))
                    {
                        return NotFound();
                    }
                    else
                    {
                        throw;
                    }
                }
                return RedirectToAction(nameof(Index));
            }
            return View(genAI);
        }

        // GET: GenAIs/Delete/5
        public async Task<IActionResult> Delete(int? id)
        {
            if (id == null || _context.GenAI == null)
            {
                return NotFound();
            }

            var genAI = await _context.GenAI
                .FirstOrDefaultAsync(m => m.Id == id);
            if (genAI == null)
            {
                return NotFound();
            }

            return View(genAI);
        }

        // POST: GenAIs/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeleteConfirmed(int id)
        {
            if (_context.GenAI == null)
            {
                return Problem("Entity set 'ApplicationDbContext.GenAI'  is null.");
            }
            var genAI = await _context.GenAI.FindAsync(id);
            if (genAI != null)
            {
                _context.GenAI.Remove(genAI);
            }
            
            await _context.SaveChangesAsync();
            return RedirectToAction(nameof(Index));
        }

        private bool GenAIExists(int id)
        {
          return (_context.GenAI?.Any(e => e.Id == id)).GetValueOrDefault();
        }
    }
}
