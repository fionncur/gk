# Build the interpolation-geometry figures.
#   make        : PDFs and PNGs for every figure
#   make png    : PNGs only
#   make clean  : remove build products (keeps sources)
#
# Requires: TeX Live with pgf + standalone, and pdftoppm (poppler-utils).

TEXDIR  := tikz
PNGDIR  := png
DPI     ?= 800

SRCS := $(wildcard $(TEXDIR)/fig*.tex)
PDFS := $(SRCS:.tex=.pdf)
PNGS := $(patsubst $(TEXDIR)/%.tex,$(PNGDIR)/%.png,$(SRCS))

.PHONY: all png clean
all: $(PDFS) $(PNGS)
png: $(PNGS)

# every figure depends on the shared preamble
$(TEXDIR)/%.pdf: $(TEXDIR)/%.tex $(TEXDIR)/common.tex
	cd $(TEXDIR) && latexmk -pdf -interaction=nonstopmode -halt-on-error $(notdir $<)

$(PNGDIR)/%.png: $(TEXDIR)/%.pdf
	@mkdir -p $(PNGDIR)
	pdftoppm -r $(DPI) -png -singlefile $< $(basename $@)

clean:
	cd $(TEXDIR) && latexmk -C >/dev/null 2>&1 || true
	rm -f $(TEXDIR)/*.aux $(TEXDIR)/*.log $(TEXDIR)/*.fls $(TEXDIR)/*.fdb_latexmk
