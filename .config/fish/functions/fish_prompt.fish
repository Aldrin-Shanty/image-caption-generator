function fish_prompt
    set_color green
    echo -n (whoami)
    set_color normal
    echo -n "@"
    set_color blue
    echo -n (prompt_hostname)
    set_color normal
    echo -n " ["
    set_color green
    echo -n (basename (pwd))
    set_color normal
    echo -n "]: "
end
