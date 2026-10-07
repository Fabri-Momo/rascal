#!/usr/bin/env bash
# tools/sync-github.sh — pousse un instantané de l'arbre courant (HEAD)
# vers la branche main du dépôt GitHub de build.
#
# Pourquoi un snapshot et pas un push classique : l'historique GitLab
# contient des blobs > 100 Mo (anciens MSI/AppImage) que GitHub rejette.
# Le dépôt GitHub ne sert qu'aux builds/releases GitHub Actions ; il
# contient des commits instantanés chaînés, l'historique réel reste ici.
#
# Prérequis : gh authentifié (gh auth login), ou un credential helper git.
#
# Usage : tools/sync-github.sh

set -euo pipefail

GH_REPO="Fabri-Momo/rascal"
GH_URL="https://github.com/${GH_REPO}.git"

cd "$(git rev-parse --show-toplevel)"

# Garde-fou : aucun fichier suivi ne doit dépasser 100 Mo (limite GitHub)
big=$(git ls-files -z | xargs -0 du -b 2>/dev/null | awk '$1 > 104000000 {print $2}')
if [ -n "$big" ]; then
  echo "Erreur : fichiers > 100 Mo, GitHub les rejetterait :" >&2
  echo "$big" >&2
  exit 1
fi

if token=$(gh auth token 2>/dev/null); then
  AUTH_URL="https://x-access-token:${token}@github.com/${GH_REPO}.git"
  GIT_OPTS="-c credential.helper= -c core.askpass="
else
  AUTH_URL="$GH_URL"
  GIT_OPTS=""
fi

echo "Récupération de GitHub main..."
# shellcheck disable=SC2086
git $GIT_OPTS fetch "$AUTH_URL" main

tree=$(git rev-parse "HEAD^{tree}")
sha=$(git rev-parse --short HEAD)
parent=$(git rev-parse FETCH_HEAD)

if [ "$(git rev-parse 'FETCH_HEAD^{tree}')" = "$tree" ]; then
  echo "Déjà synchronisé : GitHub main pointe sur le même arbre."
  exit 0
fi

commit=$(git commit-tree "$tree" -p "$parent" -m "Sync from GitLab $sha")

echo "Push du snapshot $commit (GitLab $sha)..."
# shellcheck disable=SC2086
git $GIT_OPTS push "$AUTH_URL" "$commit:refs/heads/main"

echo "OK — GitHub main = $commit"
echo "Pour builder une release : onglet Actions de GitHub -> 'Build RASCAL releases' -> Run workflow"
