.PHONY: talk_slides

talk_slides:
	reveal-md slides.md --style blood --static docs --disable-auto-open
