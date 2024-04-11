$(document).foundation();

// Push footer to bottom of page
function pushFooter() {
    let footer = $("#footer");
    let pos = footer.position();
    let height = $(window).height();
    height = height - pos.top;
    height = height - (footer.height() + 4 * 16 );
    if (height > 0) {
        footer.css({
            'margin-top': height + 'px'
        });
    }
}
$(window).bind("load", pushFooter());

// Scroll to top
jQuery(document).ready(function() {
    let offset = 220;
    let duration = 500;
    jQuery(window).scroll(function() {
        if (jQuery(this).scrollTop() > offset) {
            jQuery('.back-to-top').fadeIn(duration);
        } else {
            jQuery('.back-to-top').fadeOut(duration);
        }
    });

    jQuery('.back-to-top').click(function(event) {
        event.preventDefault();
        jQuery('html, body').animate({scrollTop: 0}, duration);
        return false;
    });
});

function flashNoty(type, message) {
    new Noty({
        text: message,
        type: type,
        layout: 'topRight',
        timeout: 4000,
        theme: 'relax',
        queue: 'global',
        animation: {
            open: 'animated fadeInRight faster',
            close: 'animated fadeOutRight faster',
        }
    }).show();
}
