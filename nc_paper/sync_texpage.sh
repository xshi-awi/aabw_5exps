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
    echo "remote revision/ now holds:"
    ls -la revision/ | tail -n +2
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
