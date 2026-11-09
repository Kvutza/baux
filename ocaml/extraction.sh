#!/bin/sh
set -eu
dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
exec "$dir/tool.sh" extraction "$@"
