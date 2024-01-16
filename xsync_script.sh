#!/bin/bash

export $PATH=/home/zhicai/bin:$PATH
remote_list=("zhicai.9" "zhicai.10")
# 
root_dir=/home/zhicai
# syn_config_list=(".config" ".bashrc" ".profile" ".vimrc" ".vim" ".viminfo" ".zsh" ".zshrc" "Repos" ".oh-my-zsh" ".tmux.conf" ".tmux" ".bash_history" ".local" ".zsh_history" ".kaggle" "conda" ".condarc" "bin" ".shell.pre-oh-my-zsh")
syn_config_list=()

other_file_list=("/data/zhicai/cache/huggingface/diffusers")
#
# Function for parallel synchronization
parallel_syn() {
    remote="$1"
    echo "Syncing files for $remote..."

    # Sync syn_config_list
    for config_file in "${syn_config_list[@]}"; do
        file_path="$root_dir/$config_file"
        echo "syn $file_path"
        xsync "$file_path" -remote "$remote"
    done

    for other_file in "${other_file_list[@]}"; do
        echo "syn $other_file"
        xsync "$other_file" -remote "$remote"
    done

    echo "Done syncing files for $remote."
}

# Loop through remote servers and start parallel tasks
for remote in "${remote_list[@]}"; do
    parallel_syn "$remote" &
done

# Wait for all background tasks to finish
wait

echo "All servers processed."
