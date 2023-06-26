// makes sure the whole site is loaded
$(window).on("load", function () {
  // will first fade out the loading animation
  $("#status").fadeOut();
  // will fade out the whole DIV that covers the website.
  $("#preloader").delay(500).fadeOut("slow");
});
