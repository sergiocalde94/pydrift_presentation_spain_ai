.PHONY: talk_slides

talk_slides:
	reveal-md slides.md --theme blood --css css/social-circles.min.css,css/resize-emoji.css --assets-dir custom_assets --preprocessor js/emoji2svgimg.js --static docs --disable-auto-open
	mkdir docs/custom_assets/fonts
	cp fonts/* docs/custom_assets/fonts

