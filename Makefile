.PHONY: talk_slides

talk_slides:
	reveal-md slides.md --preprocessor js/emoji2svgimg.js --css css/social-circles.min.css,css/resize-emoji.css --theme blood --static docs --disable-auto-open

