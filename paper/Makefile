
TEX = pdflatex -shell-escape -interaction=nonstopmode -file-line-error
RM = /bin/rm

all : ebf.pdf bayesian-forests.pdf

baysian-forestis.pdf : bayesian-forests.tex
	$(TEX) bayesian-forests.tex

ebf.pdf : ebf.tex 
	$(TEX) ebf.tex

clean:
	$(RM)  *aux *log *nav *out *pdf *snm *toc *vrb
