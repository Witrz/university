$(document).ready(function () {
    var images = $(".image-container");
    console.log($(images));

    for (var i = 0; i < $(images).length; i += 1) {
        var img = $(images[i]).children("img");
        var img_url = $(img).attr('src');
        img.remove();
        $(images[i]).css({"max-height": "100%", "min-height": "150px", "background-image": "url(" + img_url + ")", "background-repeat": "repeat-x", "background-size": "contain"});
    }

    var model_container = document.getElementById("model-container");
    var items = model_container.getElementsByTagName("section");

    for (var i = 0; i < items.length; i += 1) {
        if (i % 2 == 0) {
            items[i].style.background = "linear-gradient(blue, white)";
        } else {
            items[i].style.background = "linear-gradient(orange, white)";
        }
    }
});








