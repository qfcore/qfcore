#!/usr/bin/env bash
# Install the header tree of a pinned Boost release into DEST, so that
# DEST/boost/version.hpp exists. Used by cibuildwheel (before-all) inside the
# manylinux containers and on macOS runners; setup.py finds it via BOOST_INCLUDE.
set -euo pipefail
dest="${1:?usage: install_boost_headers.sh DEST}"
version="1.90.0"
sha256="e848446c6fec62d8a96b44ed7352238b3de040b8b9facd4d6963b32f541e00f5"
if [ -f "$dest/boost/version.hpp" ]; then
  echo "Boost headers already present in $dest"
  exit 0
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
url="https://github.com/boostorg/boost/releases/download/boost-${version}/boost-${version}-b2-nodocs.tar.gz"
curl -fsSL --retry 5 --retry-delay 5 -o "$tmp/boost.tar.gz" "$url"
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$tmp/boost.tar.gz" | cut -d' ' -f1)"
else
  actual="$(shasum -a 256 "$tmp/boost.tar.gz" | cut -d' ' -f1)"
fi
if [ "$actual" != "$sha256" ]; then
  echo "Boost archive checksum mismatch: $actual" >&2
  exit 1
fi
tar -xzf "$tmp/boost.tar.gz" -C "$tmp" "boost-${version}/boost"
mkdir -p "$dest"
rm -rf "$dest/boost"
mv "$tmp/boost-${version}/boost" "$dest/boost"
echo "Boost ${version} headers installed in $dest"
