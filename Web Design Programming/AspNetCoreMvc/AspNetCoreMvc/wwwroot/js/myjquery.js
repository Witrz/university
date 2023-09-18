$(document).ready(function() {

    var headers = $('header');
    for (var i = 0; i < $(headers).length; i++) {
        if(i % 2 == 0) {
            $(headers[i]).css('color', 'navy');
        }
        else{
            $(headers[i]).css('color', 'green');
        }
    }




    $("section").mouseover(function () {
        // get its original background colour
        var bgColour = $(this).css('background-color');
        // parse the bgColour to red, green, blue, alpha
        var rgbaValues = bgColour.match(/\d+/g);
        var red = parseInt(rgbaValues[0]);
        var green = parseInt(rgbaValues[1]);
        var blue = parseInt(rgbaValues[2]);
        var alpha = parseFloat(rgbaValues[3]);
        // change alpha from its current value to 1
        alpha = 1;
        // construct the new color string with modified components
        var newBgColour = 'rgba(' + red + ', ' + green + ', ' + blue + ', ' + alpha + ')';
        // Apply the new background colour to the current section
        $(this).css('background-color', newBgColour);
        });

    $("section").mouseout(function () {
        // get its original background colour
        var bgColour = $(this).css('background-color');
        // parse the bgColour to red, green, blue, alpha
        var rgbaValues = bgColour.match(/\d+/g);
        var red = parseInt(rgbaValues[0]);
        var green = parseInt(rgbaValues[1]);
        var blue = parseInt(rgbaValues[2]);
        var alpha = parseFloat(rgbaValues[3]);
        // change alpha from its current value to 1
        alpha = 0.5;
        // construct the new color string with modified components
        var newBgColour = 'rgba(' + red + ', ' + green + ', ' + blue + ', ' + alpha + ')';
        // Apply the new background colour to the current section
        $(this).css('background-color', newBgColour);
    });

});