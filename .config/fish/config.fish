if status is-interactive
    # -----[ Aliases as Abbreviations ]-----
    abbr ls "eza -l --icons --group-directories-first --git"
    abbr grep "grep --color=auto"
    abbr adam "setxkbmap -option caps:swapescape"
    abbr ramiel "setxkbmap -option"  
    # -----[ Show Fetch on Startup ]-----
    catnap
end
set -Ux NLTK_DATA $HOME/.local/share/nltk_data
