#!/bin/bash

# Remote SSH credentials
USER="hli488"
HOST="submit1.chtc.wisc.edu"
REMOTE_DIR=$1
LOCAL_DIR="../logs"
HOME_DIR=".."


# Read the number of iterations from the remote file
NUM_ITERATIONS=$(ssh "$USER@$HOST" "grep 'queue' $REMOTE_DIR/src/submit.sub | awk '{print \$2}'")
echo "Detect Number of Iterations: $NUM_ITERATIONS"

# Loop for the extracted number of iterations
for i in $(seq 0 $((NUM_ITERATIONS - 1)))
do
    FILE_NAME="${i}_logs.tar.gz"
    echo "Copying $FILE_NAME from $HOST"
    scp "$USER@$HOST:$REMOTE_DIR/$FILE_NAME" "$HOME_DIR"

    # Check if file was copied successfully
    if [ -f "$HOME_DIR/$FILE_NAME" ]; then
        echo "Unzipping $FILE_NAME"
        # tar -xzf "$HOME_DIR/$FILE_NAME" -C "$HOME_DIR"
        tar -Pxzf "$HOME_DIR/$FILE_NAME" -C "$HOME_DIR"

        echo "Deleting $FILE_NAME"
        rm "$HOME_DIR/$FILE_NAME"
    else
        echo "Failed to copy $FILE_NAME"
    fi
done

echo "All files copied, unzipped, and original zips deleted in $HOME_DIR"
