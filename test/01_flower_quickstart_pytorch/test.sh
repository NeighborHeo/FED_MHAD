rand_str=$(openssl rand -base64 12 | sed 's/[^a-zA-Z0-9]//g')
echo $rand_str