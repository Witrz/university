using Microsoft.AspNetCore.Mvc.ModelBinding.Validation;
using System.ComponentModel.DataAnnotations;
using System.Xml.Linq;

namespace FinalAssignment.Models
{
    public class GenAI
    {
        public int Id { get; set; }
        [Display(Name = "Gen AI Name")]
        public string GenAIName { get; set; }
        [Display(Name = "Summary")]
        public string Summary { get; set; }

        [Display(Name = "Image File")]
        [ValidateNever]
        public string ImageFilename { get; set; }

        [Display(Name = "Anchor Link")]
        [ValidateNever]
        public string AnchorLink { get; set; }
        [Display(Name = "Like")]
        public int Like { get; set; }
    }
}
