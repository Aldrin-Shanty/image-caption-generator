function venvs
    set -l ENV_DIR "$HOME/Documents/Environments"
    
    if not test -d "$ENV_DIR"
        echo "❌ Environment directory not found!"
        return 1
    end
    
    # List only directories that contain 'bin/activate.fish'
    set -l ENVS (find "$ENV_DIR" -mindepth 1 -maxdepth 1 -type d -exec test -f "{}/bin/activate.fish" \; -print | xargs -n1 basename)
    
    if test (count $ENVS) -eq 0
        echo "❌ No virtual environments found!"
        return 1
    end
    
    echo "Available virtual environments:"
    for i in (seq (count $ENVS))
        echo "$i) $ENVS[$i]"
    end
    
    echo -n "Choose an environment (1-"(count $ENVS)"): "
    read choice
    
    if string match -qr '^[0-9]+$' -- "$choice"
        if test "$choice" -ge 1 -a "$choice" -le (count $ENVS)
            set -l SELECTED_VENV "$ENV_DIR/$ENVS[$choice]/bin/activate.fish"
            echo "✅ Activating: $ENVS[$choice]..."
            source "$SELECTED_VENV"
        else
            echo "❌ Invalid choice."
        end
    else
        echo "❌ Invalid input."
    end
end
