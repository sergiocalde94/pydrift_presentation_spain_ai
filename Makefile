.PHONY: talk_slides

talk_slides:
	reveal-md slides.md --theme blood --static docs --disable-auto-open
