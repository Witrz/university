using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;
using AspNetCoreMvc.Data;
using AspNetCoreMvc.Models;
using Microsoft.AspNetCore.Authorization;
using System.Data;
using Humanizer;
using Microsoft.AspNetCore.Mvc.RazorPages;
using NuGet.Configuration;
using NuGet.Protocol.Plugins;
using System.Drawing;
using System.Runtime.Intrinsics.X86;
using System.Security.Principal;

namespace AspNetCoreMvc.Controllers
{
    public class DiscussionForumsController : Controller
    {
        private readonly ApplicationDbContext _context;

        public DiscussionForumsController(ApplicationDbContext context)
        {
            _context = context;
        }

        public async Task<IActionResult> IncreaseLike(int? id)
        {
            if (id == null)
            {
                return NotFound();
            }
            var discussionForum = await _context.DiscussionForum.FindAsync(id);
            if (discussionForum == null)
            {
                return NotFound();
            }
            if (ModelState.IsValid)
            {
                try
                {
                    discussionForum.Like++;
                    _context.Update(discussionForum);
                    await _context.SaveChangesAsync();
                }
                catch (DbUpdateConcurrencyException)
                {
                    if (!DiscussionForumExists(discussionForum.Id))
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

        // GET: DiscussionForums
        [Authorize]
        public async Task<IActionResult> Index()
        {
              return _context.DiscussionForum != null ? 
                          View(await _context.DiscussionForum.ToListAsync()) :
                          Problem("Entity set 'ApplicationDbContext.DiscussionForum'  is null.");
        }

        // GET: DiscussionForums/Details/5
        public async Task<IActionResult> Details(int? id)
        {
            if (id == null || _context.DiscussionForum == null)
            {
                return NotFound();
            }

            var discussionForum = await _context.DiscussionForum
                .FirstOrDefaultAsync(m => m.Id == id);
            if (discussionForum == null)
            {
                return NotFound();
            }

            return View(discussionForum);
        }

        // GET: DiscussionForums/Create
        [Authorize(Roles = "admin, user")]
        public IActionResult Create()
        {
            return View();
        }

        // POST: DiscussionForums/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [Authorize(Roles = "admin, user")]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Create([Bind("Id,PostDate,UserName,TopicTitle,MessageContent")] DiscussionForum discussionForum)
        {
            if (ModelState.IsValid)
            {
                _context.Add(discussionForum);
                await _context.SaveChangesAsync();
                return RedirectToAction(nameof(Index));
            }
           
            return View(discussionForum);
        }

        // GET: DiscussionForums/Edit/5
        [Authorize(Roles = "admin")]
        public async Task<IActionResult> Edit(int? id)
        {
            if (id == null || _context.DiscussionForum == null)
            {
                return NotFound();
            }

            var discussionForum = await _context.DiscussionForum.FindAsync(id);
            if (discussionForum == null)
            {
                return NotFound();
            }
            return View(discussionForum);
        }

        // POST: DiscussionForums/Edit/5
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [Authorize(Roles = "admin")]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Edit(int id, [Bind("Id,PostDate,UserName,TopicTitle,MessageContent,Like")] DiscussionForum discussionForum)
        {
            if (id != discussionForum.Id)
            {
                return NotFound();
            }

            if (ModelState.IsValid)
            {
                try
                {
                    _context.Update(discussionForum);
                    await _context.SaveChangesAsync();
                }
                catch (DbUpdateConcurrencyException)
                {
                    if (!DiscussionForumExists(discussionForum.Id))
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
            return View(discussionForum);
        }

        // GET: DiscussionForums/Delete/5
        [Authorize(Roles = "admin")]
        public async Task<IActionResult> Delete(int? id)
        {
            if (id == null || _context.DiscussionForum == null)
            {
                return NotFound();
            }

            var discussionForum = await _context.DiscussionForum
                .FirstOrDefaultAsync(m => m.Id == id);
            if (discussionForum == null)
            {
                return NotFound();
            }

            return View(discussionForum);
        }

        // POST: DiscussionForums/Delete/5
        [HttpPost, ActionName("Delete")]
        [Authorize(Roles = "admin")]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeleteConfirmed(int id)
        {
            if (_context.DiscussionForum == null)
            {
                return Problem("Entity set 'ApplicationDbContext.DiscussionForum'  is null.");
            }
            var discussionForum = await _context.DiscussionForum.FindAsync(id);
            if (discussionForum != null)
            {
                _context.DiscussionForum.Remove(discussionForum);
            }
            
            await _context.SaveChangesAsync();
            return RedirectToAction(nameof(Index));
        }

        private bool DiscussionForumExists(int id)
        {
          return (_context.DiscussionForum?.Any(e => e.Id == id)).GetValueOrDefault();
        }
    }
}
