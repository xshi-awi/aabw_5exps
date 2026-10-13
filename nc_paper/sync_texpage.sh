#!/bin/bash
# Push the current revision to the TexPage project, or pull collaborators' edits.
#
# TexPage's git remote is keyed by USER id, not project id, and the repository
# holds the one project at its root. The project-id URL is rejected with
# "No permission to access this project", which reads like an auth failure but
# is not one.
#
# The token lives in ~/.config/texpage_token and is never written into a file
# that git tracks: ~/.claude is itself under version control.
#
#   ./sync_texpage.sh push    rebuild the revision folder and push it
#   ./sync_texpage.sh pull    fetch into /tmp and report what changed
set -euo pipefail

USER_ID=3b0f38dc-f5ba-46c4-9187-05a527f27be7
TOKEN_FILE=~/.config/texpage_token
WORK=/tmp/claude-texpage
NC=/work/ba1066/a270064/cc_projects/aabw_5exps/nc_paper

[ -f "$TOKEN_FILE" ] || { echo "no token at $TOKEN_FILE"; exit 1; }
TOKEN=$(cat "$TOKEN_FILE")
URL="https://git:${TOKEN}@git.texpage.com/${USER_ID}.git"
hide() { sed "s/${TOKEN}/***/g"; }

mkdir -p "$WORK"
if [ -d "$WORK/repo/.git" ]; then
    git -C "$WORK/repo" fetch -q "$URL" main 2>&1 | hide
    git -C "$WORK/repo" reset -q --hard FETCH_HEAD
else
    git clone -q "$URL" "$WORK/repo" 2>&1 | hide
fi
cd "$WORK/repo"
git config user.name "Xiaoxu Shi"
git config user.email "OhickselroyBcEW@tvstar.com"

# Assemble the folder that TexPage should hold, into $1. Used twice: once by
# the push guard to see what the remote *would* become, and once for real.
build_revision() {
    local dst=$1
    mkdir -p "$dst/figures" "$dst/letter_figs"
    cp "$NC/build/revised.tex" "$NC/build/ref.bib" \
       "$NC/build/sn-jnl.cls" "$NC/build/sn-nature.bst" "$dst/"
    cp "$NC/SUBMISSION/04_response_letter.tex" "$dst/"
    # the Reviewer 1 draft is the document under active revision
    for f in 04_response_letter_R1.tex 04_response_letter_R1.pdf \
             response_letter_R1_1013.docx; do
        [ -f "$NC/SUBMISSION/$f" ] && cp "$NC/SUBMISSION/$f" "$dst/"
    done
    [ -f "$NC/texpage_README.md" ] && cp "$NC/texpage_README.md" "$dst/README.md"
    # only the figures the manuscript actually cites, not all 52 in build/
    grep -o 'includegraphics\[[^]]*\]{[^}]*}' "$NC/build/revised.tex" \
        | sed 's/.*{\(.*\)}/\1/' | sort -u | while read -r f; do
        cp "$NC/build/$f" "$dst/figures/" 2>/dev/null || true
    done
    cp "$NC/SUBMISSION/letter_figs/"*.png "$dst/letter_figs/"
}

case "${1:-push}" in
pull)
    # show the whole repo: Xiaoxu adds folders of his own (e.g. "original
    # submission/"), and listing only revision/ hides them
    echo "remote holds:"
    for d in */; do
        printf '  %-24s %s files, %s\n' "$d" \
            "$(git ls-tree -r --name-only HEAD -- "$d" | wc -l)" \
            "$(du -sh "$d" 2>/dev/null | cut -f1)"
    done
    echo
    echo "edits made in the web editor since the last push from here:"
    git log --format='  %h  %s  (%ar)' -5 --grep="Updates from TeXPage" || echo "  none"
    echo
    echo "differences against the local copies:"
    for f in revised.tex 04_response_letter.tex 04_response_letter_R1.tex; do
        local_f="$NC/build/$f"
        [ -f "$local_f" ] || local_f="$NC/SUBMISSION/$f"
        if [ -f "$local_f" ] && ! diff -q "revision/$f" "$local_f" >/dev/null 2>&1; then
            echo "  CHANGED: $f"
            diff -u "$local_f" "revision/$f" | head -40 || true
        else
            echo "  same: $f"
        fi
    done
    echo
    echo "working copy is at $WORK/repo"
    ;;
push)
    # Xiaoxu edits the .tex directly in the TexPage web editor. Rebuilding
    # revision/ from the local copy does `rm -rf revision` first, so a push
    # does not merge his edits, it deletes them.
    #
    # The earlier version of this guard compared commit TIMESTAMPS, which was
    # useless: after any push from here, the local commit is the newest one, so
    # the check passed even though the web copy still held edits that had never
    # been folded into RESPONSE_LETTER.md. It also offered `push-force` as an
    # escape hatch, and the escape hatch got used.
    #
    # Compare CONTENT instead. If the remote .tex differs from what this script
    # would write, an edit exists on the web that is not in the local source,
    # whatever the commit dates say.
    # Xiaoxu edits response_letter_R1_1013.docx directly. Anything he wrote that
    # is not in RESPONSE_LETTER.md is destroyed by the rebuild, so check the
    # REMOTE copy before building anything.
    if [ -f revision/response_letter_R1_1013.docx ]; then
        if ! (cd "$NC" && python check_docx_merged.py \
              "$WORK/repo/revision/response_letter_R1_1013.docx"); then
            echo
            echo "Merge those into nc_paper/RESPONSE_LETTER.md, rebuild, then push."
            exit 1
        fi
    fi

    STAGE=$(mktemp -d)
    build_revision "$STAGE/revision"
    DIRTY=""
    for f in revised.tex 04_response_letter.tex 04_response_letter_R1.tex; do
        if [ -f "revision/$f" ] && \
           ! diff -q -b -B <(sed 's/[[:space:]]*$//' "revision/$f") \
                           <(sed 's/[[:space:]]*$//' "$STAGE/revision/$f") >/dev/null 2>&1; then
            DIRTY="$DIRTY $f"
        fi
    done
    # files Xiaoxu added on the web that the rebuild would not recreate
    EXTRA=$(cd revision 2>/dev/null && find . -type f | sed 's|^\./||' | sort > "$STAGE/remote.txt"
            cd "$STAGE/revision" && find . -type f | sed 's|^\./||' | sort > "$STAGE/local.txt"
            comm -23 "$STAGE/remote.txt" "$STAGE/local.txt")
    REMOTE_HASH=$(cat revision/revised.tex revision/04_response_letter.tex 2>/dev/null \
                  | md5sum | cut -c1-12)
    REVIEWED=""
    PREV=""
    for a in "$@"; do
        [ "$PREV" = "--reviewed" ] && REVIEWED=$a
        PREV=$a
    done
    if [ -n "$REVIEWED" ] && [ "$REVIEWED" = "$REMOTE_HASH" ]; then
        echo "remote state $REMOTE_HASH acknowledged as reviewed; proceeding"
        DIRTY=""
        EXTRA=""
    fi
    if [ -n "$DIRTY" ] || [ -n "$EXTRA" ]; then
        echo "The web copy differs from what this script would write."
        echo
        for f in $DIRTY; do
            echo "  $f differs.  -  what this push would write"
            echo "                  +  what the remote holds now"
            diff -u -b -B <(sed 's/[[:space:]]*$//' "$STAGE/revision/$f") \
                          <(sed 's/[[:space:]]*$//' "revision/$f") \
                | grep -E '^[-+][^-+]' | head -20 | cut -c1-240 | sed 's/^/    /' || true
            echo
        done
        if [ -n "$EXTRA" ]; then
            echo "  files only on the web:"
            echo "$EXTRA" | sed 's/^/    /'
            echo
        fi
        cat <<'MSG'
Pushing would delete these. They must go into the generator source first,
because the .tex is regenerated from nc_paper/RESPONSE_LETTER.md on every
build: editing the .tex alone is undone by the next run of md_to_tex.py.

  1. read the diff above
  2. make the same change in nc_paper/RESPONSE_LETTER.md (or build/revised.tex)
  3. rebuild, then push again

There is deliberately no --force. If the difference is a local edit of your own
that simply has not reached the remote yet, acknowledge THIS EXACT remote state
and push again:

  ./sync_texpage.sh push "message" --reviewed RHASH

The hash names the remote content you just read above, so it stops working the
moment anything changes on the remote. It is not a force flag: it cannot
discard an edit you have not seen.
MSG
        echo "  RHASH for the state shown above: $(cat revision/revised.tex revision/04_response_letter.tex 2>/dev/null | md5sum | cut -c1-12)"
        rm -rf "$STAGE"
        exit 1
    fi
    rm -rf "$STAGE"
    ;&
do_push)
    STAGE_KEEP=$(mktemp -d)
    KEPT=$(cd revision 2>/dev/null && find . -maxdepth 1 -type f \
           \( -name '*.docx' -o -name '*.txt' \) \
           ! -name '05_response_letter*.docx' \
           ! -name 'response_letter_R1_1013.docx' -printf '%f\n' 2>/dev/null || true)
    for f in $KEPT; do cp "revision/$f" "$STAGE_KEEP/" 2>/dev/null || true; done
    [ -n "$KEPT" ] && echo "keeping web uploads: $(echo $KEPT | tr '\n' ' ')"
    rm -rf revision
    build_revision revision
    # preserve anything uploaded through the web editor that the rebuild does
    # not recreate, such as an annotated docx
    if [ -d "$STAGE_KEEP" ]; then
        cp -n "$STAGE_KEEP"/* revision/ 2>/dev/null || true
    fi

    git add -A revision
    if git diff --cached --quiet; then
        echo "nothing changed"
        exit 0
    fi
    git commit -q -m "${2:-Update revision from local working copy}"
    git push -q "$URL" main 2>&1 | hide
    echo "pushed $(du -sh revision | cut -f1) to TexPage"

    # If a browser tab had the file open, TexPage may write that stale buffer
    # back a little later and undo this push. Re-fetch and compare, so the
    # answer is observed rather than assumed.
    echo "verifying (the web editor can write a stale buffer back)..."
    PUSHED_LETTER=$(md5sum revision/04_response_letter.tex | cut -d' ' -f1)
    PUSHED_MS=$(md5sum revision/revised.tex | cut -d' ' -f1)
    for _ in 1 2 3; do
        sleep 20
        git fetch -q "$URL" main 2>&1 | hide
        git reset -q --hard FETCH_HEAD
        NOW_LETTER=$(md5sum revision/04_response_letter.tex 2>/dev/null | cut -d' ' -f1)
        NOW_MS=$(md5sum revision/revised.tex 2>/dev/null | cut -d' ' -f1)
        if [ "$NOW_LETTER" != "$PUSHED_LETTER" ] || [ "$NOW_MS" != "$PUSHED_MS" ]; then
            echo
            echo "WARNING: the remote changed after the push."
            git log --format='  %h  %s  (%ar)' -2 -- revision
            echo "  Most likely an open browser tab wrote its buffer back."
            echo "  Close the TexPage tab, then run: ./sync_texpage.sh pull"
            exit 1
        fi
    done
    echo "verified: the remote still matches what was pushed"
    ;;
*)
    echo "usage: $0 [push|pull]"; exit 1 ;;
esac
