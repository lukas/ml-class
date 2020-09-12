#! /bin/sh
# Generates the lukas-code.txt file from the repo.
# Feel free to modify the TARGET_DIRectory of this script
# and point it at your own code!
# Note that this script is not compatible with Windows.

TARGET_DIR=..
find $TARGET_DIR | grep "\.py$" | grep -v "wandb" | xargs cat >> lukascode.txt
