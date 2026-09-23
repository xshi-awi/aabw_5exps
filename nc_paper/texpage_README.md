# Nature Communications revision

Working copy of the revision of *Glacial–interglacial regime shift in Southern
Ocean dense water formation*, for collaborative editing. The submitted version
is unchanged elsewhere in the project; everything here is new.

## What is in this folder

| File | What it is |
|---|---|
| `revised.tex` | The revised manuscript. Compile this one. |
| `04_response_letter.tex` | Point-by-point response to the four reviewer reports. |
| `figures/` | The 22 figures the manuscript cites. |
| `letter_figs/` | The 20 figures the response letter shows, as PNG. |
| `ref.bib` | Bibliography. |
| `sn-jnl.cls`, `sn-nature.bst` | Nature class and bibliography style, unmodified. |

## Compiling

Both documents use `sn-jnl.cls`, so set the compiler to **pdfLaTeX** and run
the usual pdflatex → bibtex → pdflatex → pdflatex sequence. The manuscript is
44 pages and cites around 90 references, so the first build takes a while; if
the compile times out, build `revised.tex` alone rather than both documents.

The response letter has no bibliography and needs only two pdflatex passes.

## Conventions in the response letter

Three styles, following the journal's own example:

- reviewer comments in plain black
- our replies in **blue bold**
- text quoted from the revised manuscript in ***blue bold italic***

Figures are numbered per reviewer, R1.1 … R1.8 for Reviewer 1, R2.1 … R2.5 for
Reviewer 2 and R3.1 … R3.7 for Reviewer 3. The same underlying figure appears
more than once where several reviewers raised the same point, each time with its
own number.

## Two copies, and which one wins

This folder is a view of a local working copy that also holds the analysis
scripts, the tracked-changes build and the checks that keep the letter and the
manuscript consistent. Edits made here sync back to git automatically, and
Xiaoxu folds them into the local copy before pushing again.

That means edits made here are safe as long as he pulls before pushing, which
the sync script now enforces. If you make a substantial change, say so, so it
does not sit here unnoticed.

## Editing here

Please edit the `.tex` files directly. Two things to keep in mind.

The response letter is generated from a markdown source held outside this
project, so edits made here need to be carried back by hand. Tell Xiaoxu what
you changed, or leave the change as a comment, rather than assuming it will
propagate.

Quotations of the manuscript inside the letter are extracted programmatically
from `revised.tex`, so that the letter cannot claim wording the manuscript does
not contain. If you change a passage in the manuscript that the letter quotes,
the quotation needs updating too.

## Still outstanding

Acknowledgements and funding text, and the Zenodo DOI for the data deposit.
Both are marked in `revised.tex` and need Xiaoxu to supply them.
