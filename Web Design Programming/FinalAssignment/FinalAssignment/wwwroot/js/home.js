var sections = document.querySelectorAll('section');

// Get width and height of the viewport
const viewportWidth = window.innerWidth;
const viewportHeight = window.innerHeight;
// Function to check if the difference between the height of the viewport
// and the top of the elem element is 200 pixels or greater
function isElementInViewport(elem) {
    var rect = elem.getBoundingClientRect();
    var buffer = 10;
    if (elem.id == "s1") {
        return (rect.top >= 0 + buffer);
    } else {
        return ((viewportHeight - rect.bottom) >= 0 && rect.top >= 0);
    }

}

// Write the scroll event handler function
// Your code to be executed when scrolling happens
function onScroll() {
    for (var i = 0; i < sections.length; i++) {
        console.log(sections[i].id);
        if (sections[i].id != "s6") {
            if (isElementInViewport(sections[i])) {
                sections[i].style.opacity = "1"
                sections[i].style.transition = "opacity 3.0s"
            } else {
                sections[i].style.opacity = "0"
                sections[i].style.transition = "opacity 3.0s"
            }
        }
    }
}

// Add the onScroll function as event handler for
// the scroll event on the viewport
window.addEventListener("scroll", onScroll);