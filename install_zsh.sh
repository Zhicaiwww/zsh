#!/bin/bash

if [ "$1" = "clean" ];
then
  echo "Removing ZSH config"
  rm -rf ~/.zsh
  rm ~/.zshrc
  rm ~/.p10k.zsh
  chsh -s $(which bash)
  exit
fi

# # ZSH check
# if ! type zsh &> /dev/null
# then
#     echo "ZSH could not be found, ask ROOT to install it by:"
#     echo "      sudo apt-get install zsh"
#     exit
# fi

# CONDA init
if type conda &> /dev/null
then
    conda init zsh
fi

# DIR creating
touch ~/.zshrc
if [ ! -d ~/.zsh ]; then
  mkdir -p ~/.zsh;
fi

# ALIAS setting
echo "alias ls='ls --color'" >> ~/.zshrc
echo "alias py='python'" >> ~/.zshrc

# PLUGINS
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/.zsh/powerlevel10k
status=$?
if [ $status -eq 0 ]; then
  echo "Pull p10k theme"
else
  echo ">Failed"
  exit
fi
echo 'source ~/.zsh/powerlevel10k/powerlevel10k.zsh-theme' >>~/.zshrc
echo '[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh' >>~/.zshrc

git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.zsh/plugins/zsh-syntax-highlighting
status=$?
if [ $status -eq 0 ]; then
  echo "Pull syntax-highlighting theme"
else
  echo ">Failed"
  exit
fi
echo "source ~/.zsh/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ~/.zshrc

git clone https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/plugins/zsh-autosuggestions
status=$?
if [ $status -eq 0 ]; then
  echo "Pull auto-suggestions theme"
else
  echo ">Failed"
  exit
fi
echo "source ~/.zsh/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ~/.zshrc

# RUN 
echo "[[ -e ~/.profile ]] && emulate sh -c 'source ~/.profile'" >> ~/.zshrc

# ENTER
chsh -s $(which zsh)
zsh
