// Please see documentation at https://docs.microsoft.com/aspnet/core/client-side/bundling-and-minification
// for details on configuring this project to bundle and minify static web assets.

// Write your JavaScript code.



var divs = document.querySelectorAll("div");
if (divs != null) {
    for (var i = 0; i < divs.length; i++) {
        if (divs[i].classList.contains("container")) {
            divs[i].classList.add("container-fluid");
            divs[i].classList.remove("container");
        }
    }
}

var headers = document.querySelectorAll("nav");
if (headers != null) {
    for (var i = 0; i < headers.length; i++) {
        headers[i].classList.remove("bg-white");
        headers[i].classList.remove("navbar-light");
        headers[i].classList.remove("border-bottom");
        headers[i].classList.add("navbar-dark");
        headers[i].classList.remove("mb-3");
        headers[i].classList.add("bg-dark");
        headers[i].classList.add("container-fluid");
        var header_list = headers[i].getElementsByTagName('ul');
        var brand_link = headers[i].getElementsByClassName('navbar-brand');
        header_list[0].innerHTML = "<li b-60p78gjs8i='' class='nav-item'><a class='nav-link text-light' href='/'>Home</a></li><li b-60p78gjs8i='' class='nav-item'><a class='nav-link text-light' href='/Home/Jobs'> Jobs </a></li><li b-60p78gjs8i='' class='nav-item'><a class='nav-link text-light' href='/GenAIs'> GenAI Sites </a></li><li b-60p78gjs8i='' class='nav-item'><a class='nav-link text-light' href='/Home/Contact'> Contact </a></li>";
        brand_link[0].textContent = "AI";
    }
}

var footers = document.querySelectorAll("footer");
if (footers != null) {
    for (var i = 0; i < footers.length; i++) {
        footers[i].classList.remove("border-top");
        footers[i].classList.remove("footer");
        footers[i].classList.remove("text-muted");
        footers[i].classList.add("container-fluid");
        footers[i].innerHTML = "<div id='footer' class='row container-fluid'><div class='col-md-3 col-sm-6 col-xs-12 d-flex flex-column justify-content-around'><a href='/'>Home</a><a href='/GenAIs'>Gen AI Sites</a></div ><div class='col-md-3 col-sm-6 col-xs-12 d-flex flex-column justify-content-around'><a href='/Home/Jobs'>Jobs</a><a href='/Home/Contact'>Contact</a></div><div class='col-md-3 col-sm-6 col-xs-12 d-flex flex-column justify-content-around'><a href='/Home/Contact'>About Us</a><a href='/Home/Contact'>Copyright Info</a></div><div class='col-md-3 col-sm-6 col-xs-12 d-flex flex-column justify-content-around'><p>Follow Us</p><div ><i class='fa fa-google'></i><i class='fa fa-youtube'></i><i class='fa fa-twitter'></i><i class='fa fa-facebook'></i><i class='fa fa-linkedin'></i></div></div></div>";
    }
}
