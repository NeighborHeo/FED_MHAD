#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Generate a random experiment key (characters and numbers only 48 characters long)
experiment_key=$(openssl rand -base64 48 | sed 's/[^a-zA-Z0-9]//g')
echo $experiment_key

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('~/.data', download=True)"

echo "Starting server"
python server.py --experiment_key "$experiment_key" &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 1`; do
    echo "Starting client $i"
    python client.py --index "$i" --experiment_key "$experiment_key" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
