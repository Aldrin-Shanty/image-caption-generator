function delete-venv
    set -l ENV_DIR "$HOME/Documents/Environments"
    
    if test -z "$argv[1]"
        echo "❌ Please specify a name: delete-venv <env_name>"
        return 1
    end
    
    set -l VENV_NAME "$argv[1]"
    set -l VENV_PATH "$ENV_DIR/$VENV_NAME"
    
    if test -d "$VENV_PATH"
        rm -rf "$VENV_PATH"
        echo "✅ Virtual environment '$VENV_NAME' deleted."
    else
        echo "❌ Virtual environment '$VENV_NAME' not found!"
    end
end
