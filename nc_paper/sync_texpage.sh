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
    for f in revised.tex 04_response_letter.tex; do
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
    # Xiaoxu edits on the TexPage web editor, and those edits sync back into
    # this git repo automatically. Rebuilding revision/ from the local copy
    # would delete them outright, not merely overwrite the files it happens to
    # regenerate, so refuse to push over anything that arrived since the last
    # push from here.
    LAST=$(git log -1 --format=%H --author="Xiaoxu Shi" --grep="" -- revision 2>/dev/null || true)
    INCOMING=$(git log --format=%h --grep="Updates from TeXPage" -1 -- revision 2>/dev/null || true)
    if [ -n "$INCOMING" ]; then
        NEWER=$(git log --format=%h "${INCOMING}..HEAD" --author="Xiaoxu Shi" -- revision 2>/dev/null | wc -l)
        if [ "$NEWER" -eq 0 ]; then
            echo "The web editor has changed revision/ since this script last pushed:"
            git log --format='  %h  %s  (%ar)' -3 --grep="Updates from TeXPage" -- revision
            echo
            echo "Pushing now would delete those edits. Look at them first:"
            echo "  ./sync_texpage.sh pull"
            echo
            echo "Once they are folded into the local copy, push with:"
            echo "  ./sync_texpage.sh push-force \"message\""
            exit 1
        fi
    fi
    ;&
push-force)
    rm -rf revision
    mkdir -p revision/figures revision/letter_figs
    cp "$NC/build/revised.tex" "$NC/build/ref.bib" \
       "$NC/build/sn-jnl.cls" "$NC/build/sn-nature.bst" revision/
    cp "$NC/SUBMISSION/04_response_letter.tex" revision/
    [ -f "$NC/texpage_README.md" ] && cp "$NC/texpage_README.md" revision/README.md

    # only the figures the manuscript actually cites, not all 52 in build/
    grep -o 'includegraphics\[[^]]*\]{[^}]*}' "$NC/build/revised.tex" \
        | sed 's/.*{\(.*\)}/\1/' | sort -u | while read -r f; do
        cp "$NC/build/$f" revision/figures/ 2>/dev/null || true
    done
    cp "$NC/SUBMISSION/letter_figs/"*.png revision/letter_figs/

    git add -A revision
    if git diff --cached --quiet; then
        echo "nothing changed"
        exit 0
    fi
    git commit -q -m "${2:-Update revision from local working copy}"
    git push -q "$URL" main 2>&1 | hide
    echo "pushed $(du -sh revision | cut -f1) to TexPage"
    ;;
*)
    echo "usage: $0 [push|pull]"; exit 1 ;;
esac
