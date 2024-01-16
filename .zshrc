# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
# Initialization code that may require console input (password prompts, [y/n]
# confirmations, etc.) must go above this block; everything else may go below.
source ~/.bashrc
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

plugins=(
  git
  zsh-syntax-highlighting
  zsh-autosuggestions
  zsh-autocomplete
)

#plugins+=(git vi-mode)
# INSERT_MODE_INDICATOR="%F{yellow}+%f"


#>>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/data/zhicai/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/zhicai/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/data/zhicai/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/data/zhicai/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

alias ls='ls --color'
alias py='python'
export TERM="xterm-256color"
source ~/.zsh/powerlevel10k/powerlevel10k.zsh-theme
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh
source /home/zhicai/.zsh/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
source /home/zhicai/.zsh/plugins/zsh-vi-mode/zsh-vi-mode.plugin.zsh
source /home/zhicai/Repos/marlonrichert/zsh-autocomplete/zsh-autocomplete.plugin.zsh

export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
export PATH="/home/zhicai/bin:$PATH"
clear
clash() {
    export http_proxy=http://localhost:8890
    export https_proxy=http://localhost:8890
}

unset_clash() {
    unset http_proxy
    unset https_proxy
}
conda activate ldm
source ~/.ssh/.sever_zshrc.sh
