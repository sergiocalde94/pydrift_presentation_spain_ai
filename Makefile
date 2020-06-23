.PHONY: talk_slides

talk_slides:
	reveal-md slides.md --static docs --disable-auto-open
