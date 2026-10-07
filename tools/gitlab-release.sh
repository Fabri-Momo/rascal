#!/usr/bin/env bash
# tools/gitlab-release.sh VERSION — publie sur GitLab la release déjà
# construite sur GitHub : tag VERSION sur main, assets dans le Package
# Registry, et Release GitLab.
#
# Prérequis :
#   - gh authentifié (gh auth login) pour télécharger les assets GitHub
#   - GITLAB_TOKEN : token API GitLab (scope api) — export GITLAB_TOKEN=...
#     (créé sur gitlab.huma-num.fr -> Settings -> Access Tokens)
#
# Usage : tools/gitlab-release.sh 1.8.9

set -euo pipefail

VERSION="${1:?Usage: $0 VERSION (ex: 1.8.9)}"
GH_REPO="Fabri-Momo/rascal"
GITLAB_HOST="${GITLAB_HOST:-https://gitlab.huma-num.fr}"
PROJECT_ID="${GITLAB_PROJECT_ID:-fmonna%2Frascal}"
PACKAGE_NAME="rascal"

: "${GITLAB_TOKEN:?export GITLAB_TOKEN=<token api GitLab>}"

cd "$(git rev-parse --show-toplevel)"

echo "==> 1/4 Tag $VERSION sur GitLab (HEAD de main local)"
git tag -f "$VERSION" HEAD
git push -f origin "refs/tags/$VERSION"

echo "==> 2/4 Téléchargement des assets de la release GitHub v$VERSION"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if ! gh release download "v$VERSION" -R "$GH_REPO" -D "$tmp" -p '*' 2>/dev/null; then
  echo "Release GitHub v$VERSION introuvable — lancez d'abord le workflow" >&2
  echo "'Build RASCAL releases' sur github.com/${GH_REPO}." >&2
  exit 1
fi
ls -lh "$tmp"

echo "==> 3/4 Upload vers le Package Registry GitLab"
links="["
first=1
for file in "$tmp"/*; do
  filename=$(basename "$file")
  url="${GITLAB_HOST}/api/v4/projects/${PROJECT_ID}/packages/generic/${PACKAGE_NAME}/${VERSION}/${filename}"
  echo "  $filename"
  curl --fail --silent --show-error \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    --upload-file "$file" \
    "$url" -o /dev/null
  [ "$first" = 1 ] || links+=","
  first=0
  links+="{\"name\": \"$filename\", \"url\": \"$url\"}"
done
links+="]"

echo "==> 4/4 Création (ou mise à jour) de la release GitLab"
data=$(LINKS="$links" VERSION="$VERSION" python -c '
import json, os
print(json.dumps({
  "name": f"RASCAL {os.environ[\"VERSION\"]}",
  "tag_name": os.environ["VERSION"],
  "description": "Automated RASCAL release built by GitHub Actions.",
  "assets": {"links": json.loads(os.environ["LINKS"])}
}))')

code=$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --request POST \
  --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
  --header "Content-Type: application/json" \
  --data "$data" \
  "${GITLAB_HOST}/api/v4/projects/${PROJECT_ID}/releases")

if [ "$code" = 200 ] || [ "$code" = 201 ]; then
  echo "Release GitLab $VERSION créée."
else
  echo "POST /releases -> HTTP $code, tentative de mise à jour (PUT)..."
  curl --fail --silent --show-error \
    --request PUT \
    --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    --header "Content-Type: application/json" \
    --data "$data" \
    "${GITLAB_HOST}/api/v4/projects/${PROJECT_ID}/releases/${VERSION}" -o /dev/null
  echo "Release GitLab $VERSION mise à jour."
fi
