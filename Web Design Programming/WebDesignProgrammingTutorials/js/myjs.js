var s1 = document.getElementById("heading");
console.log(s1);
var s2 = document.querySelector('#heading');
console.log(s2);
var s3 = document.getElementsByTagName('section');
console.log('getElementsByTagName: ', s3)
var s4 = document.querySelector('section');
console.log('querySelector: ', s4)
var s5 = document.querySelectorAll('section');
console.log('querySelectorAll: ', s5)

var sections = document.querySelectorAll('section');
if (sections.length > 0) {
    var lastIndex = sections.length - 1;
    var lastSection = sections[lastIndex];
    // Add Canberra image to this last section
    lastSection.innerHTML +=
        "<figure>" +
        " <img src='images/canberra.jpg' />" +
        " <figcaption>Canberra</figcaption>" +
        "</figure>";
}

// Implement a function for adding an image to a container element
function addImage(container, imageFilename, imageCaption) {
    container.innerHTML +=
        "<figure class='col-md-4 col-sm-6 col-12' >" +
        " <img src='images/" + imageFilename + "' />" +
        " <figcaption>" + imageCaption + "</figcaption>" +
        "</figure>";
}

// Adding multiple images using a loop with the addImage function above
// Create a new section element
var newSection = document.createElement('section');
// Add the Bootstrap row class to this section to create 4x3 grid
newSection.className = 'row';
newSection.style = 'padding: 20px 0px 20px 0px;';
// Add 12 images to this new section using the addImage function
//for (var i = 0; i < 12; i++) {
//    var imageFilename = 'ai' + (i + 1) + '.jpg';
//    var imageCaption = 'Image ' + (i + 1);
//    addImage(newSection, imageFilename, imageCaption);
//}
// Add the new section to the body of the page

newSection.onmousedown = mouseDown;
function mouseDown(obj) {
    // Add 12 images to this new section with Bootstrap columns for responsive design
    for (var i = 0; i < 12; i++) {
        var imageFilename = 'ai' + (i + 1) + '.jpg';
        var imageCaption = 'Image ' + (i + 1);
        addImage(obj.target, imageFilename, imageCaption);
    }
}

var body = document.querySelector('body');
body.appendChild(newSection);


var newStyle = "font-family:Arial; text-align:center; color:navy; background:linear-gradient(lightgray, white, gray); font - size: 30px; border - style: outset; border - width: 7px;background - size: cover; background - height: auto; ";
var currentStyle;
function changeStyle(obj) {
    currentStyle = obj.target.style;
    obj.target.style = newStyle;
}
function changeStyleBack(obj) {
    obj.target.style = currentStyle;
}
var sections = document.querySelectorAll('section');
for (var i = 0; i < sections.length; i++) {
    sections[i].onmouseenter = changeStyle;
    sections[i].onmouseleave = changeStyleBack;
}

// Add CSS class
function addClass(obj) {
    obj.target.classList.add("oxygen");
}
// Remove CSS class
function removeClass(obj) {
    obj.target.classList.remove("oxygen");
}
var sections = document.querySelectorAll('section');
for (var i = 0; i < sections.length; i++) {
    sections[i].onmouseover = addClass;
    sections[i].onmouseout = removeClass;
}


// Get width and height of the viewport
const viewportWidth = window.innerWidth;
const viewportHeight = window.innerHeight;
// Function to check if the difference between the height of the viewport
// and the top of the elem element is 200 pixels or greater
function isElementEnteringViewport(elem) {
    var rect = elem.getBoundingClientRect();
    return ( // true if the difference >= 200
        (viewportHeight - rect.top) >= 200
    );
}
// Write the scroll event handler function
// Your code to be executed when scrolling happens
function onScroll() {
    for (var i = 0; i < sections.length; i++) {
        if (isElementEnteringViewport(sections[i])) {
            sections[i].style.opacity = "1"
            sections[i].style.transition = "opacity 3.0s"
        } else {
            sections[i].style.opacity = "0"
            sections[i].style.transition = "opacity 3.0s"
        }
    }


}
// Add the onScroll function as event handler for
// the scroll event on the viewport
window.addEventListener("scroll", onScroll);
