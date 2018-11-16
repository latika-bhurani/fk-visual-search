$(function() {
    $('#side-menu').metisMenu();
});

var imageLoader = document.getElementById('filePhoto');
    imageLoader.addEventListener('change', handleImage, false);

function handleImage(e) {
    var reader = new FileReader();
    reader.onload = function (event) {
        
        $('.uploader img').attr('src',event.target.result);
    }
    reader.readAsDataURL(e.target.files[0]);
}

//Loads the correct sidebar on window load,
//collapses the sidebar on window resize.
// Sets the min-height of #page-wrapper to window size
$(function() {
    $(window).bind("load resize", function() {
        var topOffset = 50;
        var width = (this.window.innerWidth > 0) ? this.window.innerWidth : this.screen.width;
        if (width < 768) {
            $('div.navbar-collapse').addClass('collapse');
            topOffset = 100; // 2-row-menu
        } else {
            $('div.navbar-collapse').removeClass('collapse');
        }

        var height = ((this.window.innerHeight > 0) ? this.window.innerHeight : this.screen.height) - 1;
        height = height - topOffset;
        if (height < 1) height = 1;
        if (height > topOffset) {
            $("#page-wrapper").css("min-height", (height) + "px");
        }
    });

    var url = window.location;
    // var element = $('ul.nav a').filter(function() {
    //     return this.href == url;
    // }).addClass('active').parent().parent().addClass('in').parent();
    var element = $('ul.nav a').filter(function() {
        return this.href == url;
    }).addClass('active').parent();

    while (true) {
        if (element.is('li')) {
            element = element.parent().addClass('in').parent();
        } else {
            break;
        }
    }
});


// $('#name_form').submit(function(e) {
//                     e.preventDefault();
//                     var data = {};
//                     var Form = this;
//                     $.ajax({
//                         type: 'POST',
//                         url: '/api/say_name',
//                         dataType: 'json',
//                         contentType: 'application/json; charset=utf-8',
//                         data: JSON.stringify(data),
//                         context: Form,
//                         success: function(callback) {
//                             console.log(callback);
//                             // Watch out for Cross Site Scripting security issues when setting dynamic content!
//                             $(this).text('Hello ' + callback.first_name + ' ' + callback.last_name  + '!');
//                         },
//                         error: function() {
//                             $(this).html("error!");
//                         }
//                     });
//                 });